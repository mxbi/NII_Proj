# -*- coding: utf-8 -*-
"""
Created on Thu May 14 12:01:42 2020

@author: Siyu Han
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
#from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from nltk.translate.bleu_score import sentence_bleu
from nltk.metrics.distance import edit_distance
from nltk.util import ngrams

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
# This version uses k-means to separate lines

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
# This version uses width to calculate threshold

def generate_line_idx_2(one_page):
    line_index = [0]
    for char_idx in range(1, one_page.shape[0]):
        current_char = one_page.iloc[char_idx, :]
        average_center = float(one_page.iloc[line_index[-1]:char_idx, one_page.columns == 'x_center'].mean())
        average_width = float(one_page.iloc[line_index[-1]:char_idx, one_page.columns == 'Width'].mean())
        diff_center = abs(current_char['x_center'] - average_center)
        threshold = (current_char['Width'] + average_width) / 2

        if diff_center > threshold:
            line_index.append(char_idx)
        else: pass

    line_index.append(one_page.shape[0])
    return line_index

## Ignored. Internal function called by function “process_one_page”.
## Similar to v2, but add checkers. Yield better result than v2.

def generate_line_idx_3(one_page):
    line_index = [0]
    for char_idx in range(1, one_page.shape[0]):
        current_char = one_page.iloc[char_idx, :]
        average_center = float(one_page.iloc[line_index[-1]:char_idx, one_page.columns == 'x_center'].mean())
        average_width = float(one_page.iloc[line_index[-1]:char_idx, one_page.columns == 'Width'].mean())
        diff_center = abs(current_char['x_center'] - average_center)
        threshold = (current_char['Width'] + average_width) / 2

        # diff_center, threshold = generate_threshold(one_page, start_idx = line_index[-1], char_idx = char_idx)

        if diff_center > threshold:
            if one_page.shape[0] - char_idx >= 3: # check next two chars
                next_char_1 = one_page.iloc[(char_idx + 1), :]
                next_char_2 = one_page.iloc[(char_idx + 2), :]
                checker_1 = abs(next_char_1['x_center'] - average_center) > ((next_char_1['Width'] + average_width) / 2)
                checker_2 = abs(next_char_2['x_center'] - average_center) > ((next_char_2['Width'] + average_width) / 2)
                checker = all([checker_1, checker_2])
            elif one_page.shape[0] - char_idx == 2:  # check next char
                next_char = one_page.iloc[(char_idx + 1), :]
                checker = abs(next_char['x_center'] - average_center) > ((next_char['Width'] + average_width) / 2)
            else: checker = True

            if checker:
                line_index.append(char_idx)
            else: pass

        else: pass

    line_index.append(one_page.shape[0])
    return line_index

## Ignored. Internal function called by function “process_one_page”.
# This version considers line spaces of different pages in determining threshold.
# I'm trying to dynamically calculate line space for each page, rather than giving a fixed value.
# But v4 gives the worst performance among all versions.

def generate_line_idx_4(one_page):
    line_index = [0]
    line_space = pd.Series(0)
    for char_idx in range(1, one_page.shape[0]):
        mean_line_space = line_space.mean()
        current_char = one_page.iloc[char_idx, :]
        average_center = float(one_page.iloc[line_index[-1]:char_idx, one_page.columns == 'x_center'].mean())
        average_width = float(one_page.iloc[line_index[-1]:char_idx, one_page.columns == 'Width'].mean())
        diff_center = abs(current_char['x_center'] - average_center)
        threshold = ((current_char['Width'] + average_width) / 2) + mean_line_space

        # diff_center, threshold = generate_threshold(one_page, start_idx = line_index[-1], char_idx = char_idx)

        if diff_center > threshold:
            if one_page.shape[0] - char_idx >= 3: # check next two chars
                next_char_1 = one_page.iloc[(char_idx + 1), :]
                next_char_2 = one_page.iloc[(char_idx + 2), :]
                checker_1 = abs(next_char_1['x_center'] - average_center) > ((next_char_1['Width'] + average_width) / 2)
                checker_2 = abs(next_char_2['x_center'] - average_center) > ((next_char_2['Width'] + average_width) / 2)
                checker = all([checker_1, checker_2])
            elif one_page.shape[0] - char_idx == 2:  # check next char
                next_char = one_page.iloc[(char_idx + 1), :]
                checker = abs(next_char['x_center'] - average_center) > ((next_char['Width'] + average_width) / 2)
            else: checker = True

            if checker:
                line_index.append(char_idx)
                previous_char = one_page.iloc[(char_idx - 1), :]
                diff_line_space = previous_char['x_center'] - current_char['x_center']
                line_space = line_space.append(pd.Series(diff_line_space), ignore_index = True)

            else: pass

        else: pass

    line_index.append(one_page.shape[0])
    return line_index

## Ignored. Internal function called by function “process_one_page”.
# v5. Best performance among all version.
# Consider line space but using a fixed value as threshold.

def generate_line_idx_5(one_page, fixed_threshold = 5):
    line_index = [0]
    for char_idx in range(1, one_page.shape[0]):
        current_char = one_page.iloc[char_idx, :]
        average_center = float(one_page.iloc[line_index[-1]:char_idx, one_page.columns == 'x_center'].mean())
        average_width = float(one_page.iloc[line_index[-1]:char_idx, one_page.columns == 'Width'].mean())
        diff_center = abs(current_char['x_center'] - average_center)
        threshold = ((current_char['Width'] + average_width) / 2) + fixed_threshold

        # diff_center, threshold = generate_threshold(one_page, start_idx = line_index[-1], char_idx = char_idx)

        if diff_center > threshold:
            if one_page.shape[0] - char_idx >= 3:  # check next two chars
                next_char_1 = one_page.iloc[(char_idx + 1), :]
                next_char_2 = one_page.iloc[(char_idx + 2), :]
                checker_1 = abs(next_char_1['x_center'] - average_center) > ((next_char_1['Width'] + average_width) / 2)
                checker_2 = abs(next_char_2['x_center'] - average_center) > ((next_char_2['Width'] + average_width) / 2)
                checker = all([checker_1, checker_2])
            elif one_page.shape[0] - char_idx == 2:  # check next char
                next_char = one_page.iloc[(char_idx + 1), :]
                checker = abs(next_char['x_center'] - average_center) > ((next_char['Width'] + average_width) / 2)
            else:
                checker = True

            if checker:
                line_index.append(char_idx)
            else:
                pass

        else:
            pass

    line_index.append(one_page.shape[0])
    return line_index

## Ignored. Internal function called by function “check_main_part”.

def subline_checker(potential_line, first_idx, second_idx, line_space_rate = 0,
                    overlapping_rate = 0.15,
                    check_x_axis = True, check_y_axis = True):
    checker_x_axis = True
    checker_y_axis = True

    if check_x_axis:
        x_val_diff = abs(potential_line["x_center"].iloc[first_idx] - potential_line["x_center"].iloc[second_idx])
        # use average of width?
        width_threshold = (potential_line["Width"].iloc[first_idx] + potential_line["Width"].iloc[second_idx]) / 2
        checker_x_axis = x_val_diff > (width_threshold * (1 + line_space_rate))

    if check_y_axis:
        y_val_diff = abs(potential_line["y_center"].iloc[first_idx] - potential_line["y_center"].iloc[second_idx])
        height_threshold = (potential_line["Height"].iloc[first_idx] + potential_line["Height"].iloc[second_idx]) / 2
        checker_y_axis = y_val_diff < (height_threshold * (1 - overlapping_rate))

    return all([checker_x_axis, checker_y_axis])

## Ignored. Internal function called by function “process_one_page”.
# Scan the main line part and check where the subline starts.

def check_main_part(potential_line, line_space_rate = 0, overlapping_rate = 0.15):
    row_number = potential_line.shape[0]
    for idx in range(row_number - 1):
        subline_checker_main = subline_checker(potential_line = potential_line,
                                            first_idx = idx, second_idx = idx + 1,
                                            line_space_rate = line_space_rate, overlapping_rate = overlapping_rate,
                                            check_x_axis = True, check_y_axis = True)

        if subline_checker_main:
            return idx

    return row_number

## Ignored. Internal function called by function “process_one_page”.
# Scan the subline part and check where the subline ends.
# argument "space_threshold" is for the previous version. Not used now.

def check_subline_part(potential_line, space_threshold = 20):
    char_length = potential_line.shape[0]
    subline_1_idx = [int(potential_line["X"].iloc[0:2].values.argmax())]
    subline_2_idx = [int(not subline_1_idx[0])]
    
    if char_length > 2:
        latestChar_subline_1 = potential_line.iloc[subline_1_idx[0]:(subline_1_idx[0] + 1),:]
        latestChar_subline_2 = potential_line.iloc[subline_2_idx[0]:(subline_2_idx[0] + 1),:]
        # latestChar_subline_1 = subline_1_idx[0]
        # latestChar_subline_2 = subline_2_idx[0]

        for idx in range(2, char_length):
            diff_subline_1 = abs(potential_line["x_center"].iloc[idx] - float(latestChar_subline_1["x_center"]))
            diff_subline_2 = abs(potential_line["x_center"].iloc[idx] - float(latestChar_subline_2["x_center"]))
            width_threshold_1 = (potential_line["Width"].iloc[idx] + float(latestChar_subline_1["Width"])) / 2
            width_threshold_2 = (potential_line["Width"].iloc[idx] + float(latestChar_subline_2["Width"])) / 2

            subline_checker_1 = diff_subline_1 < width_threshold_1
            subline_checker_2 = diff_subline_2 < width_threshold_2
            end_of_subline = all([subline_checker_1, subline_checker_2])
            # end_of_subline = min(diff_subline_1, diff_subline_2) > space_threshold

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

def process_one_page(input_formatted_page, line_space_rate = 0, overlapping_rate = 0.15,
                     irregular_layout = True, only_return_performance = False, reference_page = None,
                     detect_line_version = 5):

    # calculate centers
    input_formatted_page["x_center"] = input_formatted_page["X"] + (input_formatted_page["Width"] / 2)
    input_formatted_page["y_center"] = input_formatted_page["Y"] + (input_formatted_page["Height"] / 2)
    one_page = input_formatted_page.sort_values(by = ['x_center'], ascending = False)
    one_page.reset_index(inplace = True)

    # select one version for line separation
    if detect_line_version == 1:
        line_index = generate_line_idx(one_page)
    elif detect_line_version == 2:
        line_index = generate_line_idx_2(one_page)
    elif detect_line_version == 3:
        line_index = generate_line_idx_3(one_page)
    elif detect_line_version == 4:
        line_index = generate_line_idx_4(one_page)
    elif detect_line_version == 5:
        line_index = generate_line_idx_5(one_page, fixed_threshold = 5)  # 5 or 10 can give good result than 0 or 20
    else: raise ValueError('Wrong detect_line_version Value!')

    flag_subline_page = False
    output = pd.DataFrame()

    # for each line:
    for idx in range(len(line_index) - 1):
       # idx += 1
        potential_line = one_page.iloc[line_index[idx]:line_index[idx + 1], :]
       # potential_line["predicted_line_num"] = idx
        potential_line.insert(0, "predicted_line_num", idx) 
        potential_line = potential_line.sort_values(by = ['Y'], ascending = True)

        if not irregular_layout:
            output = pd.concat([output, potential_line], axis = 0)
            
        while potential_line.shape[0] > 0:
            if potential_line.shape[0] > 1:
                # Check main line part
                main_part_idx = check_main_part(potential_line, line_space_rate=line_space_rate,
                                                   overlapping_rate=overlapping_rate)

                flag_subline_page = False if main_part_idx == potential_line.shape[0] else True

                if main_part_idx > 0:
                    output = pd.concat([output, potential_line.iloc[0:main_part_idx,:]], axis = 0)
                    potential_line = potential_line.drop(potential_line.index[0:main_part_idx])
            else:
                output = pd.concat([output, potential_line], axis = 0)
                break

            # check subline part:
            if potential_line.shape[0] > 1:
                subline_idx = check_subline_part(potential_line)
                output = pd.concat([output, potential_line.iloc[subline_idx,:]], axis = 0)
                potential_line = potential_line.drop(potential_line.index[subline_idx])
            elif potential_line.shape[0] == 1:
                output = pd.concat([output, potential_line], axis = 0)
            else: break

    output.set_index('index', inplace = True)
    if output.shape[0] != input_formatted_page.shape[0]:
        raise ValueError('Characters Missing!', 'Image:', output.iloc[0]['Image'],
                         'Reference:', input_formatted_page.shape[0], 'Output:', output.shape[0])

    # result evaluation
    if only_return_performance:
        perf = evaluate_result(predicted_page = output, reference_page = reference_page)
        output = pd.concat([perf, pd.DataFrame({"subline": flag_subline_page}, 
                                               index = [input_formatted_page["Image"].iloc[0]])],
                           axis = 1)
    else:
        output = [output, flag_subline_page]

    return output

## Evaluate results:

def format_ngrams(raw_ngrams):
    ngram_list = [''.join(ngram) for ngram in raw_ngrams]
    ngram_Series = pd.Series(ngram_list)
    return ngram_Series

def evaluate_ngrams(predicted_char, reference_char, n = 2):
    predicted_ngrams = ngrams(predicted_char, n)
    reference_ngrams = ngrams(reference_char, n)

    predicted_Series = format_ngrams(predicted_ngrams)
    reference_Series = format_ngrams(reference_ngrams)

    perf_precision = predicted_Series.isin(reference_Series).sum() / predicted_Series.shape
    perf_recall    = reference_Series.isin(predicted_Series).sum() / reference_Series.shape

    perf = pd.DataFrame({"perf_precision": perf_precision, "perf_recall": perf_recall})
    perf.rename(columns = {"perf_precision": "precision_n=" + str(n),
                           "perf_recall":    "recall_n=" + str(n)}, inplace = True)
    return perf

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

        perf_ngram = pd.DataFrame({"precision_n=2": 1.0, "recall_n=2": 1.0,
                                   "precision_n=3": 1.0, "recall_n=3": 1.0,
                                   "precision_n=4": 1.0, "recall_n=4": 1.0,
                                   "precision_n=5": 1.0, "recall_n=5": 1.0,},
                                  index = [0])
    else:
        edit_norm = 1 - (edit_distance("".join(reference_char), "".join(predicted_char)) / all_char)        
        if len(reference_char) > 3:
            bleu_score = sentence_bleu([reference_char], predicted_char)
        else: bleu_score = None

        perf_ngram = pd.DataFrame()
        for n in range(2, 6):
            perf_ngram_tmp = evaluate_ngrams(predicted_char = predicted_char,
                                             reference_char = reference_char, n = n)
            perf_ngram = pd.concat([perf_ngram, perf_ngram_tmp], axis = 1)
    
    perf = pd.DataFrame({"page": predicted_page["Image"].iloc[0],
                         "all_char": all_char, "correct_char": correct_char,
                         "coverage": coverage, "edit_norm": edit_norm, 
                         "bleu_score": bleu_score}, index = perf_ngram.index)
    perf = pd.concat([perf, perf_ngram], axis = 1)
    perf.set_index('page', inplace = True)

    return perf

## Wrapper of function "process_one_page". This function can process one book each time:

def predict_book(input_book_path, shuffle_input = True, random_seed = 1, 
                 irregular_layout = True, overlapping_rate = 0.2,
                 only_return_performance = False, reference_page = None,
                 detect_line_version = 1):
    
    formatted_book = format_book(input_book_path, shuffle = shuffle_input, seed = random_seed)
    
    output_perf = pd.DataFrame()
    output_book = pd.DataFrame()
    
    for idx in tqdm(range(len(formatted_book))):       
        input_formatted_page = formatted_book[idx]
        tmp_res = process_one_page(input_formatted_page, 
                                   overlapping_rate = overlapping_rate,
                                   irregular_layout = irregular_layout,
                                   only_return_performance = only_return_performance, 
                                   reference_page = reference_page,
                                   detect_line_version = detect_line_version)
        if only_return_performance:
            output_perf = output_perf.append(tmp_res)
        else:
            tmp_perf = evaluate_result(predicted_page = tmp_res[0], reference_page = reference_page)
            tmp_perf = pd.concat([tmp_perf, pd.DataFrame({"subline": tmp_res[1]},
                                                   index=[input_formatted_page["Image"].iloc[0]])],
                               axis=1)
            output_perf = output_perf.append(tmp_perf)
            output_book = output_book.append(tmp_res[0])
            
    if only_return_performance:
        output = output_perf
    else:
        output = {"predicted_book": output_book, "performance": output_perf}
    
    return output

def predict_books(book_path_list, shuffle_input = True, random_seed = 1,
                  irregular_layout = True, overlapping_rate = 0.2,
                  only_return_performance = False, reference_page = None,
                  detect_line_version = 1):
    
    output = dict()
    book_num = len(book_path_list)
    
    for idx in range(book_num):
        print(idx + 1, "/", book_num, ":", sep = "")
        
        input_book_path = book_path_list[idx]
        book_res = predict_book(input_book_path = input_book_path, shuffle_input = shuffle_input, 
                                random_seed = random_seed, overlapping_rate = overlapping_rate, 
                                irregular_layout = irregular_layout,
                                only_return_performance = only_return_performance,
                                reference_page = reference_page, detect_line_version = detect_line_version)

        book_id = book_res.index[0].split("_")[0] if only_return_performance else book_res["performance"].index[0].split("_")[0]
        
        book_dict = {book_id: book_res}
        output.update(book_dict)
    
    return output

def draw_line(coordinate_Val, image_path, line_color, output_path):
    from PIL import Image, ImageDraw
    im = Image.open(image_path)
    d = ImageDraw.Draw(im)
    d.line(coordinate_Val, fill = line_color, width = 4)
    #im.show()
    im.save(output_path)

    return 'Completed'

def display_result(predicted_df, page_id, output_path, line_color = (0, 0, 255)):
    # image_path = 'C:\\Users\\alexh\\Desktop\\NII\\100249371\\100249371_00007_1.jpg'
    # predicted_df = books_result_v5_irregular
    # output_path = 'v5_' + '100249371_00007_1.jpg'
    from os import path
    book_id = page_id.split("_")[0]
    image_path = 'C:\\Users\\alexh\\Desktop\\NII\\' + book_id + '\\' + page_id + '.jpg'
    if not path.exists(image_path):
        raise ValueError('Wrong image_path!')
    else:
        prediction_df = predicted_df[book_id]['predicted_book']
        prediction_page = prediction_df[prediction_df['Image'] == page_id]
        prediction_page["y_center"] = prediction_page["Y"] + (prediction_page["Height"] / 2)
        coordinate_Val = list(zip(prediction_page.loc[:, 'x_center'], prediction_page.loc[:, 'y_center']))

        draw_line(coordinate_Val = coordinate_Val, image_path = image_path,
                  line_color = line_color, output_path = output_path)

    return 'Completed'
