# -*- coding: utf-8 -*-
"""
Created on Fri May 15 16:09:46 2020

@author: Siyu Han
"""

import pandas as pd
import numpy as np
from bio import pairwise2
from bio.pairwise2 import format_alignment
from glob import glob
from collections import Counter
import json

def process_page_csv(page_csv):
	csv_str = page_csv["labels"].str.split(" ").explode()
	csv_df = pd.DataFrame({"Unicode": csv_str.iloc[list(range(0, len(csv_str), 3))],
						   "x_center_csv": csv_str.iloc[list(range(1, len(csv_str), 3))],
						   "y_center_csv": csv_str.iloc[list(range(2, len(csv_str), 3))]})

	csv_df["csv_position"] = csv_df["x_center_csv"].str.cat(csv_df["y_center_csv"], sep = ",")

	csv_df["x_center_csv"] = pd.to_numeric(csv_df["x_center_csv"])
	csv_df["y_center_csv"] = pd.to_numeric(csv_df["y_center_csv"])

	tmp = csv_df.iloc[:,0].map(lambda x: x.replace("U+", "\\u"))
	tmp = tmp.map(lambda x: x.encode('UTF-8').decode('unicode-escape'))
	csv_df.insert(1, "Char", tmp)

	image_id = csv_df.index
	csv_df.insert(2, "Image", image_id)

	return csv_df


def format_csv(input_csv):

	csv_df = list()
	for idx in range(input_csv.shape[0]):
		page_csv = input_csv.iloc[idx:(idx + 1),:]
		tmp_df = process_page_csv(page_csv)
		csv_df.append(tmp_df)

	return csv_df


def format_json(input_json):
	json_df = list()
	for json_key in list(input_json.keys()):
		page_json = pd.DataFrame(input_json[json_key])
		page_json.set_axis(["X", "Y", "X_1", "Y_1", "P"], axis = 1, inplace = True)
		page_json.insert(4, "Width", page_json["X_1"] - page_json["X"])
		page_json.insert(4, "Height", page_json["Y_1"] - page_json["Y"])
		page_json["x_center"] = page_json["X"] + (page_json["Width"] / 2)
		page_json["y_center"] = page_json["Y"] + (page_json["Height"] / 2)
		page_json["x_center_rounded"] = page_json["x_center"].apply(np.floor).astype(int)
		page_json["y_center_rounded"] = page_json["y_center"].apply(np.floor).astype(int)

		page_json["json_position"] = page_json["x_center_rounded"].astype(str).str.cat(page_json["y_center_rounded"].astype(str), sep = ",")

		page_json.insert(0, "Index", json_key)
		page_json.set_index("Index", inplace = True)

		json_df.append(page_json)

	return json_df


def format_data(input_csv_path, input_json_path):

	input_csv = pd.read_csv(input_csv_path, sep = ',', header = 0, index_col = 0)
	input_json = json.load(open(input_json_path, "r"))

	csv_df = format_csv(input_csv)
	json_df = format_json(input_json)

	if len(csv_df) != len(json_df):
		raise ValueError('Pages from csv and json files cannot match!')

	concat_df = pd.DataFrame()
	for idx in range(len(csv_df)):
		page_csv = csv_df[idx]
		page_json = json_df[idx]

		csv_index = ~page_csv["csv_position"].isin(page_json["json_position"]).values
		json_index = ~page_json["json_position"].isin(page_csv["csv_position"]).values

		if any(csv_index):
			json_missing = page_csv.iloc[csv_index,:]
			raise ValueError('Pages from csv and json files cannot match!')

		if any(json_index):
			csv_missing = page_json.iloc[json_index,:]
			raise ValueError('Pages from csv and json files cannot match!')

		concat_page_df = pd.concat([page_csv[["Unicode", "Char", "Image"]],
									page_json[["X", "Y", "X_1", "Y_1", "Height", "Width"]]], axis = 1)
		concat_df = concat_df.append(concat_page_df)

	input_data_list = [pd.DataFrame(y) for x, y in concat_df.groupby('Image', as_index = True)]

	return input_data_list


def process_formatted_page(input_formatted_page, input_translate_page, input_unicode_data,
						   overlapping_rate = 0.2, space_threshold = 20,
						   irregular_layout = False):

	predicted_page = process_one_page(input_formatted_page,
									  overlapping_rate = overlapping_rate,
									  space_threshold = space_threshold,
									  irregular_layout = irregular_layout,
									  only_return_performance = False)
	predicted_page["new_Char"] = predicted_page["Char"]
	predicted_page["new_Unicode"] = predicted_page["Unicode"]
	predicted_page["Status"] = 'correct'

	candidate_str = "".join(predicted_page["Char"])
	# reference_str = "".join(input_translate_page.values.flatten())
	# reference_str = input_translate_page

	reference_str = input_translate_page.replace("〳〵", "〱")

	global_score = pairwise2.align.globalxs(reference_str, candidate_str, -1, -1, one_alignment_only = True)
#        print(format_alignment(*global_score[0]))
	new_reference = global_score[0][0]
	new_candidate = global_score[0][1]

	for char_idx in range(len(new_reference)):
		char_reference = new_reference[char_idx]
		char_candidate = new_candidate[char_idx]

		if char_reference != char_candidate:
			unicode_info_inbuilt = char_reference.encode('unicode-escape').decode("utf-8").replace("\\u", "U+").upper()
			retrieve_unicode_translate = list(input_unicode_data["char"].isin([char_reference]))

			if any(retrieve_unicode_translate):
				unicode_info_translated = input_unicode_data.loc[retrieve_unicode_translate,"Unicode"].iloc[0]
				if unicode_info_inbuilt != unicode_info_translated:
					raise ValueError('Unicode in translation data cannot match the in-built data!')

			unicode_info = unicode_info_inbuilt

			if char_reference != "-" and char_candidate != "-": # mismatch

				predicted_page.iloc[char_idx:(char_idx + 1), predicted_page.columns == "new_Char"] = char_reference
				predicted_page.iloc[char_idx:(char_idx + 1), predicted_page.columns == "new_Unicode"] = unicode_info
				if char_reference == "ハ" and char_candidate == "は":
					predicted_page.iloc[char_idx:(char_idx + 1), predicted_page.columns == "Status"] = 'ハ_は'
				else:
					predicted_page.iloc[char_idx:(char_idx + 1), predicted_page.columns == "Status"] = 'mismatch'

			if char_candidate == "-": # gap in candidate string

				current_line_num = predicted_page.iloc[(char_idx - 1):char_idx,0:1].iat[0,0]

				if char_idx < predicted_page.shape[0]:
					next_line_num = predicted_page.iloc[(char_idx):(char_idx + 1),0:1].iat[0,0]
					new_line_num = current_line_num if current_line_num == next_line_num else (current_line_num + next_line_num) / 2
				else:
					new_line_num = current_line_num

				new_line = pd.DataFrame({"predicted_line_num": new_line_num, "Unicode": None,
										 "Char": None, "Image": predicted_page.iloc[0:1, predicted_page.columns == "Image"].values.flatten(),
										 "X": None, "Y": None, "X_1": None, "Y_1": None,
										 "Height": None, "Width": None, "x_center": None,
										 "new_Char": char_reference,
										 "new_Unicode": unicode_info, "Status": 'gap'}).set_index("Image", drop = False)
				predicted_page = pd.concat([predicted_page.iloc[:char_idx], new_line, predicted_page.iloc[char_idx:]])

	# examine the output
	# predicted_page['change_idx'] = (predicted_page['Status'] != predicted_page['Status'].shift()).cumsum()
	# predicted_page['check_flag'] = predicted_page['Status'] != 0
	#
	# predicted_page_changeIdx = predicted_page['change_idx'][predicted_page['check_flag']]
	# changeIdx = Counter(predicted_page_changeIdx)

	predicted_page.reset_index(inplace = True)
	check_char_idx = 0

	while check_char_idx < predicted_page.shape[0]:
		if predicted_page['Status'].iloc[check_char_idx] == 'correct':
			check_char_idx += 1
			# print(check_char_idx)
		else:
			char_error_idx = []
			while (check_char_idx < predicted_page.shape[0]) and (predicted_page['Status'].iloc[check_char_idx] != 'correct'):
				char_error_idx.append(check_char_idx)
				check_char_idx += 1

			if len(char_error_idx) > 1:
				# print("TRUE")
				candidate_chars = predicted_page['Char'].iloc[char_error_idx]
				reference_chars = predicted_page['new_Char'].iloc[char_error_idx]

				if (sum(candidate_chars == 'は') == sum(reference_chars == 'ハ')) and (sum(candidate_chars == 'は') > 0):
					reference_chars.loc[reference_chars == 'ハ'] = 'は'

				if all(candidate_chars.isin(reference_chars)):
					predicted_page.loc[char_error_idx,'Status'] = 'self_corrected' # mismatch but probability corrected
				else:
					if len(char_error_idx) > 4:
						predicted_page.loc[char_error_idx,'Status'] = 'big_mismatch' # big mismatch
			else:
				pass # blank

	return predicted_page

input_csv_path = "C:/Users/alexh/Desktop/NII/sampledata/L000021.csv"
input_json_path = "C:/Users/alexh/Desktop/NII/sampledata/L000021.json"
input_translate_path = "C:/Users/alexh/Desktop/NII/sampledata/text/002.txt"
input_unicode_path = "C:/Users/alexh/Desktop/NII/sampledata/unicode_translation.csv"
input_translate_folder = "C:/Users/alexh/Desktop/NII/sampledata/text/*.txt"

# input_translate_page = pd.read_csv(input_translate_path, sep = '\n', header = None,
# 								   index_col = None, encoding = "utf-8", skipinitialspace=True)
input_translate_page = open(input_translate_path, "r", encoding='utf-8').readlines()
input_translate_page = "".join(input_translate_page)
input_translate_page = "".join(input_translate_page.split())

input_unicode_data = pd.read_csv(input_unicode_path, sep = ',', header = 0,
								 index_col = None, encoding = "utf-8")

input_formatted_book = format_data(input_csv_path, input_json_path)
input_formatted_book = input_formatted_book[1:28]
input_formatted_page = input_formatted_book[0]
#output_result = process_formatted_page(input_formatted_page = input_formatted_page,
#                                       input_translate_page = input_translate_page, 
#                                       input_unicode_data = input_unicode_data,
#                                       overlapping_rate = 0.2, space_threshold = 20,
#                                       irregular_layout = False, use_line_info = False)

def process_formatted_book(input_formatted_book, input_translate_folder, input_unicode_data,
						   overlapping_rate = 0.2, space_threshold = 20,
						   irregular_layout = False):

	input_translate_list = glob(input_translate_folder)

	output_result = pd.DataFrame()

	for idx in range(0, len(input_formatted_book)):

		input_formatted_page = input_formatted_book[idx]
		input_translate_path = input_translate_list[idx]
		# input_translate_page = pd.read_csv(input_translate_path, sep = '\n', header = None,
		# 								   index_col = None, encoding = "utf-8", skipinitialspace = True)
		input_translate_page = open(input_translate_path, "r", encoding = 'utf-8').readlines()
		input_translate_page = "".join(input_translate_page)
		input_translate_page = "".join(input_translate_page.split())

		# for idx_2 in range(len(input_translate_page)):
		# 	input_translate_page.iloc[idx_2] = input_translate_page.iloc[idx_2].str.strip().values

		temp_result__ = process_formatted_page(input_formatted_page = input_formatted_page,
											   input_translate_page = input_translate_page,
											   input_unicode_data = input_unicode_data,
											   overlapping_rate = overlapping_rate,
											   space_threshold = space_threshold,
											   irregular_layout = irregular_layout)

		output_result = output_result.append(temp_result__)

	return output_result

overall_book = process_formatted_book(input_formatted_book = input_formatted_book,
									  input_translate_folder = input_translate_folder,
									  input_unicode_data = input_unicode_data,
									  overlapping_rate = 0.2, space_threshold = 20,
									  irregular_layout = False)

overall_book_formatted = overall_book.loc[:,["Image", "predicted_line_num", "Char", "new_Char",
											 "Unicode", "new_Unicode", "Status",
											 "X", "Y", "X_1", "Y_1"]]

overall_book_formatted["Image"] = overall_book_formatted["Image"].map(lambda x: x.replace("L000021/", ""))
overall_book_formatted.insert(0, "Book_id", "L000021")
overall_book_formatted.rename(columns = {'predicted_line_num': 'Predicted_line_num', 'Char': 'Char_machine', 'new_Char': 'Char_human',
										 'Unicode': 'Unicode_machine', 'new_Unicode': 'Unicode_human',
										 'Status': 'Matching_type', 'X': 'X_0', 'Y': 'Y_0'}, inplace = True)

overall_book_formatted.to_csv("L000021_output_v3.csv")

overall_book_formatted = pd.read_csv("C:/Users/alexh/Desktop/NII/L000021_output_v3.csv",
                                     sep = ',', header = 0, index_col = None, encoding = "utf-8")

df_unmatch = overall_book_formatted.loc[overall_book_formatted['Matching_type'] != 'correct',:]
df_wrong = overall_book_formatted.loc[overall_book_formatted['Matching_type'].isin(['mismatch', 'gap', 'big_mismatch']),:]

df_gap = df_wrong.loc[df_wrong['Matching_type'] == 'gap']
df_mismatch = df_wrong.loc[df_wrong['Char_machine'].map(lambda x: x is not None),:]

stat_gap = df_gap['Char_human'].value_counts()
stat_gap.to_csv("stat_gap.csv")
stat_mismatch = df_mismatch['Char_human'].value_counts()
stat_mismatch.to_csv("stat_mismatch.csv")

tmp = df_mismatch.loc[df_mismatch['Char_human'] == 'ハ']
tmp_2 = tmp.loc[tmp['Char_machine'] != tmp['Char_machine'].iloc[1]]

tmp_2 = (list(overall_book_formatted.loc[overall_book_formatted['Image'] == '028.jpg', 'Char_machine']))
''.join([char for char in tmp_2 if char is not None])