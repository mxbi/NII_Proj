### Examples. Ignored.

import pickle
from glob import glob

input_book_path = "C:/Users/alexh/Desktop/NII/csv_folder/100249371_coordinate.csv"

formatted_book = format_book(input_book_path)
input_formatted_page = formatted_book[18]
tmp_res = process_one_page(input_formatted_page)

book_res_2 = predict_book(input_book_path, only_return_performance = False)

book_path_list = glob("C:/Users/alexh/Desktop/NII/csv_folder/*.csv")
books_result_v4_irregular = predict_books(book_path_list, detect_line_version = 1, irregular_layout = True, only_return_performance = False)
books_result_v5_irregular = predict_books(book_path_list, detect_line_version = 2, irregular_layout = True, only_return_performance = False)
books_result_v6_irregular = predict_books(book_path_list, detect_line_version = 'test', irregular_layout = True, only_return_performance = False)
books_result_v7_irregular = predict_books(book_path_list, detect_line_version = 4, irregular_layout = True, only_return_performance = False)

pickle.dump(books_result_v7_irregular, open("books_result_v7_irregular.pkl", "wb"))

books_res_df = pd.concat(books_result_v4_irregular)
books_res_df.to_csv("py_res_v5.csv")
books_res_df.mean(axis = 0)
books_res_df.loc[books_res_df["subline"] == True,:].mean(axis = 0)
books_res_df.loc[books_res_df["subline"] != True,:].mean(axis = 0)

books_result_v4_irregular = pickle.load(open("books_result_v4_irregular.pkl", "rb"))
books_result_v5_irregular = pickle.load(open("books_result_v5_irregular.pkl", "rb"))
books_result_v6_irregular = pickle.load(open("books_result_v6_irregular.pkl", "rb"))

tmp = books_res_df[(books_res_df['edit_norm'] < 0.7) & (books_res_df['bleu_score'] < 0.7)]

display_result(predicted_df = books_result_v4_irregular, page_id = '100249371_00011_2',
               output_path = '100249371_00011_2_v4.jpg', line_color = (0, 0, 255))

def compare_version (page_id, old_version_df, new_version_df, suffix):
    display_result(predicted_df = old_version_df, page_id = page_id,
                   output_path = page_id + '_' + suffix[0] + '.jpg', line_color = (0, 0, 255))
    display_result(predicted_df = new_version_df, page_id = page_id,
                   output_path = page_id + '_' + suffix[1] + '.jpg', line_color = (0, 0, 255))
    return "Completed"

compare_version(page_id = '100249371_00011_2', suffix = ['v4', 'v7'], old_version_df = books_result_v4_irregular, new_version_df = books_result_v7_irregular)

tmp = books_result_v4_irregular['100249371']['performance']

def format_perf(output_df):
    output_perf = pd.DataFrame()
    for Keys in output_df.keys():
        tmp_perf = output_df[Keys]['performance']
        output_perf = output_perf.append(tmp_perf)
    return output_perf

perf_tmp = format_perf(books_result_v7_irregular)
perf_tmp.mean(axis = 0)
perf_tmp.loc[perf_tmp["subline"] != True,:].mean(axis = 0)
perf_tmp.loc[perf_tmp["subline"] == True,:].mean(axis = 0)

