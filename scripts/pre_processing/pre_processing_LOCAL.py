import pysam
import vcf
from vcf.parser import _Info as VcfInfo, field_counts as vcf_field_counts
import math

CHR = 22
chr_to_num = lambda x: ''.join([c for c in x if c.isdigit()])
purity = 0.6

# Open files
normalSample = pysam.AlignmentFile("../../data/external/HG002.hs37d5.300x.bam", "rb", ignore_truncation=True)
tumorSample = pysam.AlignmentFile("../../data/external/HG001.hs37d5.300x.bam", "rb", ignore_truncation=True)

normalvcf = vcf.Reader(open("../../data/external/HG002_GRCh37_1_22_v4.1_draft_benchmark.vcf", 'r'))
normalvcf.infos['datasetsmissingcall'] = VcfInfo('datasetsmissingcall', None, 'String',
												  'Names of datasets that are missing a call or have an incorrect call at this location, and the high-confidence call is a variant',
												  None, None)

tumorvcf = vcf.Reader(open("../../data/external/HG001_GRCh37_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X-SOLID_CHROM1-X_v.3.3.2_highconf_PGandRTGphasetransfer.vcf", 'r'))
tumorvcf.infos['datasetsmissingcall'] = VcfInfo('datasetsmissingcall', None, 'String',
												  'Names of datasets that are missing a call or have an incorrect call at this location, and the high-confidence call is a variant',
												  None, None)

# normalOutput = open(snakemake.output[0], "w")
# normalOutput.write("CHR\tPOS\tREF\tALT\tLABEL\n")

# tumorOutput = open(snakemake.output[1], "w")
# tumorOutput.write("CHR\tPOS\tREF\tALT\tLABEL\n")

# Retrieve genomic region locations from BED
regions = []
bed = open("../../data/external/chr22_exons.bed", "r")
next(bed)

for line in bed:
	region_start, region_end = line.split()[1:3]
	regions.append((int(region_start), int(region_end)))


# Retrieve mutations from VCFs
normalMutations = {}

# for record in normalvcf:
# current_chr = chr_to_num(record.CHROM)
# 	if current_chr == '':
# 		break
# 	elif int(current_chr) < CHR:
# 		continue
# 	elif int(current_chr) > CHR:
# 		break

# 	if any(region_start <= record.POS <= region_end for (region_start, region_end) in regions):
# 		normalMutations[record.POS] = record.REF[0]

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
			print(record.POS)
			# 17309881
		break


# Process BAM file
for region in regions:
	for pileup_column in normalSample.pileup("{}".format(CHR), 17309880, 17309881):
		
		pos = pileup_column.pos + 1

		# TEST CONTROL
		if pos not in tumorMutations.keys():
			continue


		bases = {"A": 0, "T": 0, "C": 0, "G": 0}
		ref = 0.0
		alt = 0.0

		# Count up pileup column reads
		for pileup_read in pileup_column.pileups:
			if not pileup_read.is_del and not pileup_read.is_refskip:
				read = pileup_read.alignment.query_sequence[pileup_read.query_position]
				if read in ["A","T","C","G"]:
					bases[read] += 1

		values = list(bases.values())

		somaticVar = tumorMutations.get(pos, None)  

		if somaticVar:
			tumor_bases = {"A": 0, "T": 0, "C": 0, "G": 0}
			for tumor_pileup_column in tumorSample.pileup("{}".format(CHR), pos-1, pos):
				if tumor_pileup_column.pos == pos-1:
					for pileup_read in tumor_pileup_column.pileups:
						if not pileup_read.is_del and not pileup_read.is_refskip:
							read = pileup_read.alignment.query_sequence[pileup_read.query_position]
							if read in ["A","T","C","G"]:
								tumor_bases[read] += 1

			tumor_values = list(tumor_bases.values())

			combined_values = [math.floor((1-purity)*x) + math.ceil(purity*y) for (x,y) in zip(values, tumor_values)]

			combined_bases = {"A": combined_values[0], "T": combined_values[1], "C": combined_values[2], "G": combined_values[3]}

			tumor_ref = combined_bases[somaticVar]
			tumor_label = "SomaticVariant"

			combined_values.remove(tumor_ref)
			tumor_alt = float(max(combined_values))

			total = ref + alt

			# if total == 0:
			# 	continue

			tumor_ref = round(tumor_ref / total, 3)
			tumor_alt = round(tumor_alt / total, 3)

			tumorOutput.write("{}\t{}\t{}\t{}\t{}\n".format(CHR,pos,tumor_ref,tumor_alt,tumor_label))
		
		germlineVar = normalMutations.get(pos, None)

		if germlineVar:
			ref = bases[germlineVar]
			label = "GermlineVariant" 
		else:
			ref = float(max(values))
			label = "Normal"
			
		values.remove(ref)
		alt = float(max(values))
		
		total = ref + alt

		if total == 0:
			continue

		ref = round(ref / total, 3)
		alt = round(alt / total, 3)

		normalOutput.write("{}\t{}\t{}\t{}\t{}\n".format(CHR,pos,ref,alt,label))
		
		
		tumorOutput.write("{}\t{}\t{}\t{}\t{}\n".format(CHR,pos,ref,alt,label))