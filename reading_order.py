# -*- coding: utf-8 -*-
"""
Created on Thu May 14 12:01:42 2020

@author: Siyu Han
"""

import pandas as pd
import numpy as np
#import pickle
from tqdm import tqdm
#from glob import glob
#from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from nltk.translate.bleu_score import sentence_bleu
from nltk.metrics.distance import edit_distance

## Format input file.

def format_book(input_book_path, shuffle = True, seed = 1):
    input_data = pd.read_csv(input_book_path, sep = ',',header = 0)
    tmp = input_data.iloc[:,0].map(lambda x: x.replace("U+", "\\u"))
    tmp = tmp.map(lambda x: x.encode('UTF-8').decode('unicode-escape'))
    input_data.insert(1, "Char", tmp)
    
    if shuffle: input_data = input_data.sample(frac = 1, random_state = seed)
    
    input_data_list = [pd.DataFrame(y) for x, y in input_data.groupby('Image', as_index = True)]
        
    return input_data_list

## Ignored. Internal function called by function “process_one_page”.

def generate_line_idx(one_page):
    x_val_diff = abs(np.diff(one_page["x_center"]))

    if max(x_val_diff) <= min(one_page["Width"]):
        line_index = [0, len(x_val_diff)]
    else:
        x_val_diff = np.array(x_val_diff).reshape([-1,1])
        
       # gmm_res = GaussianMixture(n_components = 2, init_params = "random").fit(x_val_diff)
       # outlier_threshold = max(gmm_res.means_) * 3
        
        outlier_threshold = max(one_page["Width"]) * 3

        x_val_diff_cluster = x_val_diff[x_val_diff < outlier_threshold, None]
        kmeans_mod = KMeans(n_clusters = 2).fit(x_val_diff_cluster)
        kmeans_res = kmeans_mod.predict(x_val_diff_cluster)
        group_1 = x_val_diff_cluster[kmeans_res == 1]
        group_2 = x_val_diff_cluster[kmeans_res != 1]

        potential_threshold = max(min(group_1), min(group_2))
        line_index = np.append(np.where(x_val_diff >= potential_threshold)[0], len(x_val_diff)) + 1
        line_index = np.append(0, line_index)
        line_index = np.unique(line_index).tolist()

    return line_index

## Ignored. Internal function called by function “process_one_page”.

def check_main_part(potential_line, overlapping_rate):
    row_number = potential_line.shape[0]    
    for idx in range(row_number - 1):
        y_val_diff = potential_line["Y"].iloc[idx + 1] - potential_line["Y"].iloc[idx]
        overlap_checker = y_val_diff < (potential_line["Height"].iloc[idx] * (1 - overlapping_rate))
        
        if overlap_checker:
           # if row_number >= (idx + 5):
            if row_number >= (idx + 4):
                y_val_diff_sub_1 = potential_line["Y"].iloc[idx + 3] - potential_line["Y"].iloc[idx + 2]
               # y_val_diff_sub_2 = potential_line["Y"].iloc[idx + 4] - potential_line["Y"].iloc[idx + 3]
                overlap_checker_sub_1 = y_val_diff_sub_1 < (potential_line["Height"].iloc[idx + 2] * (1 - overlapping_rate))
               # overlap_checker_sub_2 = y_val_diff_sub_2 < (potential_line["Height"].iloc[idx + 3] * (1 - overlapping_rate))
                
               # if (overlap_checker_sub_1 or overlap_checker_sub_2):
                if (overlap_checker_sub_1):
                    return idx
            else:
                return idx
            
    return row_number

## Ignored. Internal function called by function “process_one_page”.

def check_subline_part(potential_line, space_threshold):
    row_number = potential_line.shape[0]
    subline_1_idx = [int(potential_line["X"].iloc[0:2].values.argmax())]
    subline_2_idx = [int(not subline_1_idx[0])]
    
    if row_number > 2:        
        latestChar_subline_1 = potential_line.iloc[subline_1_idx[0]:(subline_1_idx[0] + 1),:]
        latestChar_subline_2 = potential_line.iloc[subline_2_idx[0]:(subline_2_idx[0] + 1),:]
        
        for idx in range(2, row_number):
            diff_subline_1 = abs(potential_line["x_center"].iloc[idx] - float(latestChar_subline_1["x_center"]))
            diff_subline_2 = abs(potential_line["x_center"].iloc[idx] - float(latestChar_subline_2["x_center"]))
            
            end_of_subline = min(diff_subline_1, diff_subline_2) > space_threshold
            
            if end_of_subline:
                break
            else:
                if diff_subline_1 < diff_subline_2:
                    subline_1_idx.append(idx)
                    latestChar_subline_1 = potential_line.iloc[idx:(idx + 1),:]
                else:
                    subline_2_idx.append(idx)
                    latestChar_subline_2 = potential_line.iloc[idx:(idx + 1),:]
                    
    subline_idx = subline_1_idx + subline_2_idx
    
    return subline_idx

## Main function, process one page each time:

def process_one_page(input_formatted_page, overlapping_rate = 0.2, space_threshold = 20,
                     irregular_layout = False, only_return_performance = True, reference_page = None):
    
    input_formatted_page["x_center"] = input_formatted_page["X"] + (input_formatted_page["Width"] / 2)
    one_page = input_formatted_page.sort_values(by = ['x_center'], ascending = False)    
    
    line_index = generate_line_idx(one_page)    
    flag_subline_page = False
    output = pd.DataFrame()
    
    for idx in range(len(line_index) - 1):
       # idx += 1
        potential_line = one_page.iloc[line_index[idx]:line_index[idx + 1], :]
       # potential_line["predicted_line_num"] = idx
        potential_line.insert(0, "predicted_line_num", idx) 
        potential_line = potential_line.sort_values(by = ['Y'], ascending = True)
        
        y_val_diff = abs(np.diff(potential_line["Y"]))
        y_val_ref  = np.array(potential_line["Height"][0:len(y_val_diff)]) * (1 - overlapping_rate)
        
        flag_subline = sum(y_val_diff - y_val_ref < 0) > 3 if irregular_layout else False
        
        if not flag_subline:
            output = pd.concat([output, potential_line], axis = 0)
        else:
            flag_subline_page = True
            
            while potential_line.shape[0] > 0:                
                if potential_line.shape[0] > 1:
                    main_part_idx = check_main_part(potential_line, overlapping_rate = overlapping_rate)
                    if main_part_idx > 0:
                        output = pd.concat([output, potential_line.iloc[0:main_part_idx,:]], axis = 0)
                        potential_line = potential_line.drop(potential_line.index[0:main_part_idx])
                else:
                    output = pd.concat([output, potential_line], axis = 0)
                    break
                
                if potential_line.shape[0] > 1:
                    subline_idx = check_subline_part(potential_line, space_threshold = space_threshold)
                    output = pd.concat([output, potential_line.iloc[subline_idx,:]], axis = 0)
                    potential_line = potential_line.drop(potential_line.index[subline_idx])
                elif potential_line.shape[0] == 1:
                    output = pd.concat([output, potential_line], axis = 0)
                else: break
    
    if only_return_performance:
        perf = evaluate_result(predicted_page = output, reference_page = reference_page)
        output = pd.concat([perf, pd.DataFrame({"subline": flag_subline_page}, 
                                               index = [input_formatted_page["Image"].iloc[0]])], axis = 1)
    
    return output  

## Evaluate results:

def evaluate_result(predicted_page, reference_page = None):
    
    if reference_page is None:
        reference_page = predicted_page.sort_values(by = ['Char ID'], ascending = True)
    
    reference_char = reference_page["Char"].values
    predicted_char = predicted_page["Char"].values
    all_char = len(reference_char)
    correct_char = sum(reference_char == predicted_char)
    coverage = correct_char / all_char
    
    if coverage == 1:
        edit_norm = 1.0
        bleu_score = 1.0
    else:
        edit_norm = 1 - (edit_distance("".join(reference_char), "".join(predicted_char)) / all_char)        
        if len(reference_char) > 3:
            bleu_score = sentence_bleu([reference_char], predicted_char)
        else: bleu_score = None
    
    perf = pd.DataFrame({"all_char": all_char, "correct_char": correct_char,
                         "coverage": coverage, "edit_norm": edit_norm, 
                         "bleu_score": bleu_score}, index = [predicted_page["Image"].iloc[0]])
    
    return perf

## Wrapper of function "process_one_page". This function can process one book each time:

def predict_book(input_book_path, shuffle_input = True, random_seed = 1, 
                 irregular_layout = False, overlapping_rate = 0.2, space_threshold = 20, 
                 only_return_performance = True, reference_page = None):
    
    formatted_book = format_book(input_book_path, shuffle = shuffle_input, seed = random_seed)
    
    output_perf = pd.DataFrame()
    output_book = pd.DataFrame()
    
    for idx in tqdm(range(len(formatted_book))):       
        input_formatted_page = formatted_book[idx]
        tmp_res = process_one_page(input_formatted_page, 
                                   overlapping_rate = overlapping_rate,
                                   space_threshold = space_threshold,
                                   irregular_layout = irregular_layout,
                                   only_return_performance = only_return_performance, 
                                   reference_page = reference_page)
        if only_return_performance:
            output_perf = output_perf.append(tmp_res)
        else:
            tmp_perf = evaluate_result(predicted_page = tmp_res, reference_page = reference_page)
            output_perf = output_perf.append(tmp_perf)
            output_book = output_book.append(tmp_res)
            
    if only_return_performance:
        output = output_perf
    else:
        output = {"predicted_book": output_book, "performance": output_perf}
    
    return output

def predict_books(book_path_list, shuffle_input = True, random_seed = 1,
                  irregular_layout = False, overlapping_rate = 0.2, space_threshold = 20,
                  only_return_performance = True, reference_page = None):
    
    output = dict()
    book_num = len(book_path_list)
    
    for idx in range(book_num):
        print(idx + 1, "/", book_num, ":", sep = "")
        
        input_book_path = book_path_list[idx]
        book_res = predict_book(input_book_path = input_book_path, shuffle_input = shuffle_input, 
                                random_seed = random_seed, overlapping_rate = overlapping_rate, 
                                space_threshold = space_threshold, irregular_layout = irregular_layout,
                                only_return_performance = only_return_performance, reference_page = reference_page)

        book_id = book_res.index[0].split("_")[0] if only_return_performance else book_res["performance"].index[0].split("_")[0]
        
        book_dict = {book_id: book_res}
        output.update(book_dict)
    
    return output


### Examples. Ignored.

#input_book_path = "C:/Users/alexh/Desktop/NII/csv_folder/brsk00000_coordinate.csv"
#
#formatted_book = format_book(input_book_path)
#input_formatted_page = formatted_book[95]
#tmp_res = process_one_page(input_formatted_page)
#
#book_res_2 = predict_book(input_book_path, only_return_performance = False)
#
#book_path_list = glob("C:/Users/alexh/Desktop/NII/csv_folder/*.csv")
#books_result_v3 = predict_books(book_path_list)
#pickle.dump(books_result_v4, open("py_res_v4.pkl", "wb"))
#
#pickle.dump(books_result, open("py_res_v1.pkl", "wb"))
#
#books_res_df = pd.concat(books_result)
#books_res_df.to_csv("py_res_v1.csv")
#books_res_original_df = pd.concat(books_result_original)
#books_res_original_df.to_csv("py_res_v0.csv")
#books_res_v4_df = pd.concat(books_result_v4)
#books_res_v4_df.to_csv("py_res_v4.csv")
#
#books_res_v3 = pickle.load(open("py_res_v3.pkl", "rb"))
#books_res_v3_df = pd.concat(books_result_v3)
#books_res_v3_df.mean(axis = 0)
#books_res_v3_df.loc[books_res_v3_df["subline"] == True,:].mean(axis = 0)
#books_res_v3_df.loc[books_res_v3_df["subline"] != True,:].mean(axis = 0)
