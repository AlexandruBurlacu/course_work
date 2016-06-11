from breeds import male_weights, female_weights, male_heights, female_heights
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

fig = plt.figure()
fig.add_subplot(221)
plt.hist(male_weights, stacked = False)

fig.add_subplot(222)
plt.hist(female_weights, stacked = False)

fig.add_subplot(223)
plt.hist(male_heights, stacked = False)

fig.add_subplot(224)
plt.hist(female_heights, stacked = False)

plt.savefig('fig.jpg')