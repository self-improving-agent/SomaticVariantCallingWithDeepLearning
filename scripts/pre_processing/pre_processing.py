import pysam
import vcf
from vcf.parser import _Info as VcfInfo, field_counts as vcf_field_counts
import math
import numpy as np

CHR = snakemake.config['CHR']
purity = snakemake.config['purity']

chr_to_num = lambda x: ''.join([c for c in x if c.isdigit()])

# Open files
normalSample = pysam.AlignmentFile(snakemake.input[0], "rb", ignore_truncation=True)
tumorSample = pysam.AlignmentFile(snakemake.input[1], "rb", ignore_truncation=True)

normalvcf = vcf.Reader(open(snakemake.input[2], 'r'))
normalvcf.infos['datasetsmissingcall'] = VcfInfo('datasetsmissingcall', None, 'String',
												  'Names of datasets that are missing a call or have an incorrect call at this location, and the high-confidence call is a variant',
												  None, None)

tumorvcf = vcf.Reader(open(snakemake.input[3], 'r'))
tumorvcf.infos['datasetsmissingcall'] = VcfInfo('datasetsmissingcall', None, 'String',
												  'Names of datasets that are missing a call or have an incorrect call at this location, and the high-confidence call is a variant',
												  None, None)

normalOutput = open(snakemake.output[0], "w")
normalOutput.write("CHR\tPOS\tREF\tALT\tGAPS\tLABEL\n")

tumorOutput = open(snakemake.output[1], "w")
tumorOutput.write("CHR\tPOS\tREF\tALT\tGAPS\tLABEL\n")

# Retrieve genomic region locations from BED
regions = []
bed = open(snakemake.input[4], "r")
next(bed)

for line in bed:
	region_start, region_end = line.split()[1:3]
	regions.append((int(region_start), int(region_end)))

# Retrieve mutations from VCFs
normalMutations = {}

for record in normalvcf:
	current_chr = chr_to_num(record.CHROM)
	if current_chr == '':
		break
	elif int(current_chr) < CHR:
		continue
	elif int(current_chr) > CHR:
		break

	if any(region_start <= record.POS <= region_end for (region_start, region_end) in regions):
		normalMutations[record.POS] = record.REF[0]

tumorMutations = {}
uniqueTumorMutations = {}

for record in tumorvcf:
	current_chr = chr_to_num(record.CHROM)
	if current_chr == '':
		break
	elif int(current_chr) < CHR:
		continue
	elif int(current_chr) > CHR:
		break

	if any(region_start <= record.POS <= region_end for (region_start, region_end) in regions):
		# Extra condition to get mutations unique to tumor sample
		if record.POS not in normalMutations.keys():
			uniqueTumorMutations[record.POS] = record.REF[0]
		else:
			tumorMutations[record.POS] = record.REF[0]

# Process BAM file
for region in regions:
	skipped = []
	germline_to_copy = {}

	# Process tumor sample
	for tumor_pileup_column in tumorSample.pileup("{}".format(CHR), region[0], region[1]):
		pos = tumor_pileup_column.pos + 1
		tumor_bases = {"A": 0, "T": 0, "C": 0, "G": 0}
		tumor_gaps = 0.0

		# Count up pileup column reads
		for pileup_read in tumor_pileup_column.pileups:
			if not pileup_read.is_del and not pileup_read.is_refskip:
				read = pileup_read.alignment.query_sequence[pileup_read.query_position]
				if read in ["A","T","C","G"]:
					tumor_bases[read] += 1
			if pileup_read.indel != 0:
				tumor_gaps += 1

		tumor_values = list(tumor_bases.values())

		# Take purity % of reads from tumor reads
		tumor_values = [math.ceil(purity*base) for base in tumor_values]
		tumor_bases = {"A": tumor_values[0], "T": tumor_values[1], "C": tumor_values[2], "G": tumor_values[3]}
		tumor_gaps = math.ceil(purity*tumor_gaps)

		# Determine classes and select reads
		somaticVar = uniqueTumorMutations.get(pos, None)
		germlineVar = tumorMutations.get(pos, None)

		if somaticVar:
			tumor_ref = tumor_bases[somaticVar]
			tumor_label = "SomaticVariant"
		elif germlineVar:
			tumor_ref = tumor_bases[germlineVar]
			tumor_label = "GermlineVariant"

			germline_to_copy[pos] = {"ref": germlineVar,"values": tumor_values.copy(), "bases": tumor_bases.copy(), "gaps": tumor_gaps}
		else:
			tumor_ref = float(max(tumor_values))
			tumor_label = "Normal"

		tumor_values.remove(tumor_ref)
		tumor_alt = float(max(tumor_values))

		tumor_total = tumor_ref + tumor_alt + tumor_gaps

		if tumor_total == 0:
			skipped.append(pos)
			continue

		tumor_ref = round(tumor_ref / tumor_total, 3)
		tumor_alt = round(tumor_alt / tumor_total, 3)
		tumor_gaps = round(tumor_gaps / tumor_total, 3)

		tumorOutput.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(CHR,pos,tumor_ref,tumor_alt,tumor_gaps,tumor_label))

	# Process normal sample
	for pileup_column in normalSample.pileup("{}".format(CHR), region[0], region[1]):
		
		pos = pileup_column.pos + 1
		bases = {"A": 0, "T": 0, "C": 0, "G": 0}
		gaps = 0.0

		# Count up pileup column reads
		for pileup_read in pileup_column.pileups:
			if not pileup_read.is_del and not pileup_read.is_refskip:
				read = pileup_read.alignment.query_sequence[pileup_read.query_position]
				if read in ["A","T","C","G"]:
					bases[read] += 1
			if pileup_read.indel != 0:
				gaps += 1

		values = list(bases.values())
		
		# Take, 1-purity % of reads from normal reads
		values = [math.ceil((1-purity)*base) for base in values]
		bases = {"A": values[0], "T": values[1], "C": values[2], "G": values[3]}
		gaps = math.ceil((1-purity)*gaps)

		# Determine classes and select reads
		#germlineVar = tumorMutations.get(pos, None)
		germlineVar = germline_to_copy.get(pos, None)

		if germlineVar:
			values = germlineVar["values"]
			bases = germlineVar["bases"]
			gaps = germlineVar["gaps"]

			ref = bases[germlineVar["ref"]]
			label = "GermlineVariant"
		else:
			ref = float(max(values))
			label = "Normal"
			
		values.remove(ref)
		alt = float(max(values))

		# When the class is germline variant, reads are copied from the tumor sample with noise
		if germlineVar:
			ref = abs(ref + np.random.normal(0,0.1,1)[0])
			alt = abs(alt + np.random.normal(0,0.1,1)[0])
			gaps = abs(gaps + np.random.normal(0,0.1,1)[0])

		total = ref + alt + gaps
		
		if total == 0 or (pos in skipped):
			continue

		ref = round(ref / total, 3)
		alt = round(alt / total, 3)
		gaps = round(gaps / total, 3)

		# Record values
		normalOutput.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(CHR,pos,ref,alt,gaps,label))