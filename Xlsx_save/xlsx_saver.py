
"""
Created by Wanglong Lu on 2023/08
"""
import pandas as pd


class Xlsx_saver:

    def __init__(self,xlsx_name=None):
        self.dic_list = []
        self.col_name_list = []

        self.xlsx_name = xlsx_name
        if xlsx_name==None:
            self.xlsx_name = "xlsx_data"


    def append(self,item,col_name=None):
        if col_name == None:
            col_name = str(len(self.col_name_list))
        self.col_name_list.append(col_name)
        self.dic_list.append(item)

    def save(self):
        df = pd.DataFrame(self.dic_list)
        # Set row labels
        df.index = self.col_name_list  # Here you need to ensure that the number of row labels is the same as the number of rows in the DataFrame
        # Save to Excel including row labels
        df.to_excel(f"{self.xlsx_name}.xlsx", index=True)




if __name__ == '__main__':
    import pandas as pd

    # Example: multi-line dictionary data
    data = [
        {"fid_value": 1, "U_IDS_score": 2, "P_IDS_score": 3, "mae": 4, "psnr": 5, "ssim": 6, "lpips": 7},
        {"fid_value": 2, "U_IDS_score": 3, "P_IDS_score": 4.00005645400, "mae": 5, "psnr": 6, "ssim": 7, "lpips": 8898.34342343242321},
        # ... more lines
    ]

    # # create DataFrame
    # df = pd.DataFrame(data)
    #
    # # Set row labels
    # row_labels = ['col1', 'col2']  #Here you need to ensure that the number of row labels is the same as the number of rows in the DataFrame
    # df.index = row_labels
    #
    # # Save to Excel including row labels
    # df.to_excel("data_with_row_labels.xlsx", index=True)

    saver = Xlsx_saver("data")
    saver.append(data[0],"col1")
    saver.append(data[1],"col2")
    saver.save()
