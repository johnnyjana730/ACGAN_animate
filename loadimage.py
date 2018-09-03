
import os
SCRIPT_PATH = os.path.dirname(os.path.abspath( __file__ ))
import numpy as np
import pandas  as pd
import glob  

# def get_categories(datasetpath):
#     '''得到所有分类，文件夹名称'''
#     cat_paths = [files
#                  for files in glob(datasetpath + "/*")
#                   if isdir(files)]
#     cat_paths.sort()
#     cats = [basename(cat_path) for cat_path in cat_paths]
#     return cats

def return_label_num():
    tem_files = glob.glob(SCRIPT_PATH+'/images/*')
    return(len(tem_files))

def pictureload():  
    tem_files = glob.glob(SCRIPT_PATH+'/images/*')  
    # print('files = ',len(tem_files),'images = ', tem_files)
    pre_index = None
    k = 0
    for i in tem_files:
        x = i.split('\\')[-1]
        index =  pd.DataFrame(glob.glob(i+'/*.*'))
        index['label'] = k
        pre_index = pd.concat([index, pre_index],axis=0)
        k= k +  1
    return pre_index
