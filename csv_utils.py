import csv
from breeds import male_weights, female_weights, male_heights, female_heights, breeds

def csv_write(param, data, filename = 'breeds.csv'):
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter = ",")
        writer.writerow(param)
        for row in data:
            writer.writerow(row)

def csv_read(filename):
    data = []
    with open(filename) as file:
        reader = csv.reader(file)
        for line in reader:
            data.append(line)
    return data

if __name__ == '__main__':
    # data = zip(male_weights, female_weights, male_heights, female_heights, breeds)
    # param = ("male weight", "female weight", "male height", "female height", "breed")
    
    # csv_write(param, data)
    # csv_read('breeds.csv')
    pass