"""
    German shepherd -> male_weight: 30-40 kg; male_height: 60-65 cm; female_weight: 22-32 kg; female_height: 55-60 cm;
    Anatolian shepherd -> male_weight: 50-65 kg; male_height: 74-81 cm; female_weight: 40-55 kg; female_height: 71-79 cm;
    Australian shepherd -> male_weight: 23-29 kg; male_height: 51-58 cm; female_weight: 14-20 kg; female_height: 46-53 cm;
    Belgian shepherd -> male_weight: 25-30 kg; male_height: 60-66 cm; female_weight: 20-25 kg; female_height: 56-62 cm;
    Caucasian shepherd -> male_weight: 55-100 kg; male_height: 70-90 cm; female_weight: 45-80 kg; female_height: 65-75 cm;
    
    Bibliography:
        1) https://en.wikipedia.org/wiki/German_Shepherd
        2) https://en.wikipedia.org/wiki/Anatolian_Shepherd
        3) https://en.wikipedia.org/wiki/Australian_Shepherd
        4) https://en.wikipedia.org/wiki/Belgian_Shepherd
        5) https://en.wikipedia.org/wiki/Caucasian_Shepherd_Dog
"""

from numpy import mean, random, std, concatenate

def sigma2(rang):
    return std(rang)

def miu(rang):
    return mean(rang)


# german shepherd stats
german_shep_male_w = (30, 40)
german_shep_male_h = (60, 65)
german_shep_female_w = (22, 32) 
german_shep_female_h = (55, 60)

# anatolian shepherd stats
anatol_shep_male_w = (50, 65)
anatol_shep_male_h = (74, 81)
anatol_shep_female_w = (40, 55) 
anatol_shep_female_h = (71, 79)

# australian shepherd stats
austral_shep_male_w = (23, 29)
austral_shep_male_h = (51, 58)
austral_shep_female_w = (14, 20)
austral_shep_female_h = (46, 53)

# belgian shepherd stats
belg_shep_male_w = (25, 30)
belg_shep_male_h = (60, 66)
belg_shep_female_w = (20, 25) 
belg_shep_female_h = (56, 62)

# caucasian shepherd stats
cauc_shep_male_w = (55, 100)
cauc_shep_male_h = (70, 90)
cauc_shep_female_w = (45, 80) 
cauc_shep_female_h = (65, 75)

random.seed(0)
############################################

male_weights = concatenate(
                [sigma2(german_shep_male_w) * random.randn(1000) + miu(german_shep_male_w),
                sigma2(anatol_shep_male_w) * random.randn(1000) + miu(anatol_shep_male_w),
                sigma2(austral_shep_male_w) * random.randn(1000) + miu(austral_shep_male_w),
                sigma2(belg_shep_male_w) * random.randn(1000) + miu(belg_shep_male_w),
                sigma2(cauc_shep_male_w) * random.randn(1000) + miu(cauc_shep_male_w)]    )


female_weights = concatenate(
                [sigma2(german_shep_female_w) * random.randn(1000) + miu(german_shep_female_w),
                sigma2(anatol_shep_female_w) * random.randn(1000) + miu(anatol_shep_female_w),
                sigma2(austral_shep_female_w) * random.randn(1000) + miu(austral_shep_female_w),
                sigma2(belg_shep_female_w) * random.randn(1000) + miu(belg_shep_female_w),
                sigma2(cauc_shep_female_w) * random.randn(1000) + miu(cauc_shep_female_w) ]   )


male_heights = concatenate(
                [sigma2(german_shep_male_h) * random.randn(1000) + miu(german_shep_male_h),
                sigma2(anatol_shep_male_h) * random.randn(1000) + miu(anatol_shep_male_h),
                sigma2(austral_shep_male_h) * random.randn(1000) + miu(austral_shep_male_h),
                sigma2(belg_shep_male_h) * random.randn(1000) + miu(belg_shep_male_h),
                sigma2(cauc_shep_male_h) * random.randn(1000) + miu(cauc_shep_male_h)  ]  )


female_heights = concatenate(
                [sigma2(german_shep_female_h) * random.randn(1000) + miu(german_shep_female_h),
                sigma2(anatol_shep_female_h) * random.randn(1000) + miu(anatol_shep_female_h),
                sigma2(austral_shep_female_h) * random.randn(1000) + miu(austral_shep_female_h),
                sigma2(belg_shep_female_h) * random.randn(1000) + miu(belg_shep_female_h),
                sigma2(cauc_shep_female_h) * random.randn(1000) + miu(cauc_shep_female_h)]    )


breeds = (["german shepheard" for _ in range(1000)] + 
          ["anatolian shepherd" for _ in range(1000)] +
          ["australian shepherd" for _ in range(1000)] +
          ["belgian shepherd" for _ in range(1000)] +
          ["caucasian shepherd" for _ in range(1000)])
          

############################################

import csv

schema = lambda x, y, z, k, b: [x, y, x, k, b]

with open('breeds.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter = ",")
    writer.writerow(("male weight", "female weight", "male height", "female height", "breed"))
    for row in zip(male_weights, female_weights, male_heights, female_heights, breeds):
        writer.writerow(row) 

with open('breeds.csv') as file:
    reader = csv.reader(file)
    for line in reader:
        pass
