__author__ = 'nhan'
from numpy import array


# iterated through dataframe to get uniques dictionary
uniques = {'buying': array(['low', 'med', 'high', 'vhigh'], dtype=object),
 'class': array(['unacc', 'acc', 'vgood', 'good'], dtype=object),
 'doors': array(['2', '3', '4', '5more'], dtype=object),
 'lug_boot': array(['small', 'med', 'big'], dtype=object),
 'maint': array(['low', 'med', 'high', 'vhigh' ], dtype=object),
 'persons': array(['2', '4', 'more'], dtype=object),
 'safety': array(['low', 'med', 'high'], dtype=object)}

features = ['buying', 'safety', 'doors', 'lug_boot', 'maint', 'persons']
target_feature = ['class']

result = {'unacc' : 'red', 'acc' : 'blue', 'vgood' : 'green', 'good' : 'purple'}