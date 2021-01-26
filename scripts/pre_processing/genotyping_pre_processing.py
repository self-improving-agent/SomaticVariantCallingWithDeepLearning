import pysam
import vcf
from vcf.parser import _Info as VcfInfo, field_counts as vcf_field_counts
import math

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
			tumorMutations[record.POS] = record.REF[0]


# Process BAM file
for region in regions:
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

		somaticVar = tumorMutations.get(pos, None)

		# Mix in reads from tumor sample if there is a somatic variant recorded at the location
		if somaticVar:
			tumor_bases = {"A": 0.0, "T": 0.0, "C": 0.0, "G": 0.0}
			tumor_gaps = 0.0

			for tumor_pileup_column in tumorSample.pileup("{}".format(CHR), pos-1, pos):
				if tumor_pileup_column.pos == pos-1:
					for pileup_read in tumor_pileup_column.pileups:
						if not pileup_read.is_del and not pileup_read.is_refskip:
							read = pileup_read.alignment.query_sequence[pileup_read.query_position]
							if read in nucleotides:
								tumor_bases[read] += 1
						if pileup_read.indel != 0:
							tumor_gaps += 1

			tumor_values = list(tumor_bases.values())

			# Take purity % of reads from tumor reads, 1-purity % of reads from normal reads
			combined_values = [math.floor((1-purity)*x) + math.ceil(purity*y) for (x,y) in zip(values, tumor_values)]
			combined_bases = {"A": combined_values[0], "T": combined_values[1], "C": combined_values[2], "G": combined_values[3]}

			tumor_nuc = somaticVar
			tumor_label = "SomaticVariant"

			total = sum(tumor_values) + tumor_gaps

			a = round(tumor_values[0] / total, 3)
			t = round(tumor_values[1] / total, 3)
			c = round(tumor_values[2] / total, 3)
			g = round(tumor_values[3] / total, 3)
			tumor_gaps = round(tumor_gaps / total, 3)

			tumorOutput.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(CHR,pos,a,t,c,g,tumor_gaps,tumor_nuc,tumor_label))
		
		germlineVar = normalMutations.get(pos, None)

		if germlineVar:
			nuc = germlineVar
			label = "GermlineVariant"
		else:
			nuc = nucleotides[values.index(max(values))]
			label = "Normal"
			
		total = sum(values) + gaps

		if total == 0:
			continue

		a = round(values[0] / total, 3)
		t = round(values[1] / total, 3)
		c = round(values[2] / total, 3)
		g = round(values[3] / total, 3)
		gaps = round(gaps / total, 3)

		normalOutput.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(CHR,pos,a,t,c,g,gaps,nuc,label))
		
		if not somaticVar:
			tumorOutput.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(CHR,pos,a,t,c,g,gaps,nuc,label))