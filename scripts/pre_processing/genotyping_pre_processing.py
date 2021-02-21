import pysam
import vcf
from vcf.parser import _Info as VcfInfo, field_counts as vcf_field_counts
import math
import numpy as np

CHR = snakemake.config['CHR']
purity = snakemake.config['purity']

chr_to_num = lambda x: ''.join([c for c in x if c.isdigit()])
nucleotides = ["A","T","C","G"]

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
normalOutput.write("CHR\tPOS\tA\tT\tC\tG\tGAP\tNUC\tLABEL\n")

tumorOutput = open(snakemake.output[1], "w")
tumorOutput.write("CHR\tPOS\tA\tT\tC\tG\tGAP\tNUC\tLABEL\n")

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
			tumor_nuc = somaticVar
			tumor_label = "SomaticVariant"
		elif germlineVar:
			tumor_nuc = germlineVar
			tumor_label = "GermlineVariant"

			germline_to_copy[pos] = {"nuc": germlineVar,"values": tumor_values.copy(), "bases": tumor_bases.copy(), "gaps": tumor_gaps}
		else:
			tumor_nuc = nucleotides[tumor_values.index(max(tumor_values))]
			tumor_label = "Normal"

		tumor_total = sum(tumor_values) + tumor_gaps

		if tumor_total == 0:
			skipped.append(pos)
			continue

		tumor_a = round(tumor_values[0] / tumor_total, 3)
		tumor_t = round(tumor_values[1] / tumor_total, 3)
		tumor_c = round(tumor_values[2] / tumor_total, 3)
		tumor_g = round(tumor_values[3] / tumor_total, 3)
		tumor_gaps = round(tumor_gaps / tumor_total, 3)

		tumorOutput.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(CHR,pos,tumor_a,tumor_t,tumor_c,tumor_g,tumor_gaps,tumor_nuc,tumor_label))

	for pileup_column in normalSample.pileup("{}".format(CHR), region[0], region[1]):
		
		pos = pileup_column.pos + 1
		bases = {"A": 0.0, "T": 0.0, "C": 0.0, "G": 0.0}
		gaps = 0.0

		# Count up pileup column reads
		for pileup_read in pileup_column.pileups:
			if not pileup_read.is_del and not pileup_read.is_refskip:
				read = pileup_read.alignment.query_sequence[pileup_read.query_position]
				if read in nucleotides:
					bases[read] += 1
			if pileup_read.indel != 0:
				gaps += 1

		values = list(bases.values())

		# Take 1-purity % of reads from normal reads
		values = [math.ceil((1-purity)*base) for base in values]
		bases = {"A": values[0], "T": values[1], "C": values[2], "G": values[3]}
		gaps = math.ceil((1-purity)*gaps)

		# Determine classes and select reads
		germlineVar = germline_to_copy.get(pos, None)

		if germlineVar:
			values = germlineVar["values"]
			bases = germlineVar["bases"]
			gaps = germlineVar["gaps"]

			nuc = germlineVar["nuc"]
			label = "GermlineVariant"
		else:
			nuc = nucleotides[values.index(max(values))]
			label = "Normal"
		
		# When the class is germline variant, reads are copied from the tumor sample with noise
		if germlineVar:
			a = abs(values[0] + np.random.normal(0,0.1,1)[0])
			t = abs(values[1] + np.random.normal(0,0.1,1)[0])
			c = abs(values[2] + np.random.normal(0,0.1,1)[0])
			g = abs(values[3] + np.random.normal(0,0.1,1)[0])
			gaps = abs(gaps + np.random.normal(0,0.1,1)[0])
		else:
			a,t,c,g = values[0], values[1], values[2], values[3]

		total = a + t + c + g + gaps

		if total == 0 or (pos in skipped):
			continue

		a = round(a / total, 3)
		t = round(t / total, 3)
		c = round(c / total, 3)
		g = round(g / total, 3)
		gaps = round(gaps / total, 3)	

		normalOutput.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(CHR,pos,a,t,c,g,gaps,nuc,label))