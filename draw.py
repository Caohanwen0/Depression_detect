# import openpyxl

# wb = openpyxl.load_workbook('data/data_clean_all.xlsx')
# ws = wb.active
# serial = []
# label = []
# for col in ws['A']:
#     serial.append(col.value) 
# for col in ws['L']:
#     label.append(col.value)
# del serial[0]
# del label[0]
# texts = []
# print("  \nReading files to construct raw dataset\n   ")
# for serial_num in serial:
#     f = open('data/text_pool/' + str(serial_num) + ".txt", 'r')
#     raw = f.read()
#     texts.append(raw)
#     f.close()

# print("delete blank data")
# i = 0
# while i < len(texts):
#     if len(texts[i].strip()) == 0:
#         del texts[i]
#         del label[i]
#     i += 1

# import pandas as pd
# data = {'text': texts, 'label':label}
# df = pd.DataFrame(data)
# df.to_csv('text.csv', sep = ',', mode = 'w')

import pandas as pd
import matplotlib.pyplot as plt
df_raw = pd.read_csv('text.csv')
df_raw['text'] = df_raw['text'].astype(str) 
df_raw['length'] = df_raw['text'].apply(len)

len_df      = df_raw.groupby('length').count()
sent_length = len_df.index.tolist()
sent_freq   = len_df['text'].tolist()

# 绘制句子长度及出现频数统计图
plt.bar(sent_length, sent_freq, color="blue")
plt.title("Sentence length",)
plt.xlabel("sentence length", )
plt.ylabel("frequency", )
plt.savefig('length.jpg', bbox_inches='tight', dpi=450,) 
plt.show()

