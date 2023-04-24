import pandas
import math
df = pandas.read_csv('paper.csv', encoding='GBK')
#print(df)
len1 = len(df)
# Matching Net\cite{} &&$43.56_{\pm0.84} $ &$55.31_{\pm0.73}$ &  &  &$56.53_{\pm 0.99}$ &$63.54_{ \pm 0.85}$ & & \\

print(len(df['mini-1'][0]))
str = df['mini-1'][0]
print(str.find('±'))

print(str[0:5],str[7:])

def get_2_str(str1):
    if type(str1).__name__!='str':
        return '&'
    i = str1.find('±')

    return '&$'+str1[0:i-1]+'_{\pm'+str1[i+1:]+'}$'
for i in range(len1):
    m1=get_2_str(df['mini-1'][i])
    m5=get_2_str(df['mini-5'][i])
    t1 = get_2_str(df['tiered-1'][i])
    t5 = get_2_str(df['tiered-5'][i])
    c1 = get_2_str(df['CUB-1'][i])
    c5 = get_2_str(df['CUB-5'][i])
    ci1 = get_2_str(df['CIFAR-1'][i])
    ci5 = get_2_str(df['CIFAR-5'][i])
    print(r'%s\cite{}$_{%s%d}$ &%s%s\\'\
           %(df['Method'][i],df['Venue'][i],df['Year'][i],df['Backbone'][i],m1+m5+t1+t5+c1+c5+ci1+ci5))
    # print('$%s\cite{}_{%s%d}$ &%s &$%s_{\pm%s}$ &$%s_{\pm%s}$ &$%s_{\pm%s}$  &$%s_{\pm%s}$ &$%s_{\pm%s}$ &$%s_{\pm%s}$ &$%s_{\pm%s}$  &$%s_{\pm%s}$ \\ '\
    #       %(df['Method'][i],df['Venue'][i],df['Year'][i],df['Backbone'][i],m1[0],m1[1],m5[0],m5[1],t1[0],t1[1],t5[0],t5[1],c1[0],c1[1],c5[0],c5[1],ci1[0],ci1[1],ci5[0],ci5[1]))