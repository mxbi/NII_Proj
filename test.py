### Examples. Ignored.

import pickle
from glob import glob

input_book_path = "C:/Users/alexh/Desktop/NII/csv_folder/100249371_coordinate.csv"
input_book_path = "C:/Users/alexh/Desktop/NII/csv_folder/brsk00000_coordinate.csv"

formatted_book = format_book(input_book_path)
input_formatted_page = formatted_book[18]
input_formatted_page = formatted_book[6]
tmp_res = process_one_page(input_formatted_page)

res_kmeans = predict_book(input_book_path, detect_line_version = 1, irregular_layout = True, only_return_performance = True)

book_path_list = glob("C:/Users/alexh/Desktop/NII/csv_folder/*.csv")
books_result_v4_irregular = predict_books(book_path_list, detect_line_version = 1, irregular_layout = True, only_return_performance = True)
books_result_v5_irregular = predict_books(book_path_list, detect_line_version = 2, irregular_layout = True, only_return_performance = False)
books_result_v6_irregular = predict_books(book_path_list, detect_line_version = 3, irregular_layout = True, only_return_performance = False)
books_result_v7_irregular = predict_books(book_path_list, detect_line_version = 4, irregular_layout = True, only_return_performance = False)
books_result_v8_irregular = predict_books(book_path_list, detect_line_version = 5, irregular_layout = True, only_return_performance = False)


books_result_v9_irregular = predict_books(book_path_list, detect_line_version = 5, process_subline_version = 2, irregular_layout = True, only_return_performance = False)
pickle.dump(books_result_v9_irregular, open("books_result_v9_irregular.pkl", "wb"))
pickle.dump(books_result_v8_irregular, open("books_result_v8_irregular.pkl", "wb"))
pickle.dump(books_result_v7_irregular, open("books_result_v7_irregular.pkl", "wb"))

books_res_df = pd.concat(books_result_v4_irregular)
books_res_df.to_csv("py_res_v5.csv")
books_res_df.mean(axis = 0)
books_res_df.loc[books_res_df["subline"] == True,:].mean(axis = 0)
books_res_df.loc[books_res_df["subline"] != True,:].mean(axis = 0)

books_result_v4_irregular = pickle.load(open("books_result_v4_irregular.pkl", "rb"))
books_result_v5_irregular = pickle.load(open("books_result_v5_irregular.pkl", "rb"))
books_result_v6_irregular = pickle.load(open("books_result_v6_irregular.pkl", "rb"))
books_result_v8_irregular = pickle.load(open("books_result_v8_irregular.pkl", "rb"))

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

def format_prediction(output_df):
    output_pred = pd.DataFrame()
    for Keys in output_df.keys():
        tmp_pred = output_df[Keys]['predicted_book']
        output_pred = output_pred.append(tmp_pred)
    return output_pred

perf_tmp = format_perf(books_result_v9_irregular)
pred_tmp = format_prediction(books_result_v9_irregular)

perf_tmp.to_csv("C:/Users/alexh/Desktop/NII/NII_Evaluation/NII_EvaluationReport/predicted_res_v9.csv", index = True)
pred_tmp.to_csv("C:/Users/alexh/Desktop/NII/NII_Evaluation/NII_EvaluationReport/predicted_book_v9.csv", index = True)


def predict_book(formatted_book,
                 irregular_layout=True, overlapping_rate=0.2, space_threshold=20,
                 only_return_performance=False, reference_page=None,
                 detect_line_version=1):

    output_book = pd.DataFrame()
    for idx in tqdm(range(len(formatted_book))):
        # print(idx)
        input_formatted_page = formatted_book[idx]
        tmp_res = process_one_page(input_formatted_page,
                                   overlapping_rate=overlapping_rate,
                                   space_threshold=space_threshold,
                                   irregular_layout=irregular_layout,
                                   only_return_performance=only_return_performance,
                                   reference_page=reference_page,
                                   detect_line_version=detect_line_version)
        output_book = output_book.append(tmp_res[0])

    return output_book

book_res_v1_kmeans = predict_book(formatted_book, only_return_performance = False, detect_line_version = 1)
book_res_v3_rules  = predict_book(formatted_book, only_return_performance = False, detect_line_version = 3)

input_formatted_page = formatted_data[2]
book_res_v1_kmeans.to_csv("C:/Users/alexh/Desktop/NII/newTest/predicted_res_v1_kmeans.csv", index = False)
book_res_v3_rules.to_csv("C:/Users/alexh/Desktop/NII/newTest/predicted_res_v3_allRules.csv", index = False)

v1_res = book_res_v1_kmeans['Char'].values
v3_res = book_res_v3_rules['Char'].values
sum(v1_res == v3_res) / len(v3_res)

sum_val = 0.0
for i in range(0, len(csv_df)):
    sum_val = sum_val + csv_df[i].shape[0]

raw_df = pd.DataFrame()
for i in range(0, len(formatted_data)):
    raw_df = pd.concat([raw_df, formatted_data[i]], axis=0)

v1_count = book_res_v1_kmeans['predicted_book']['Image'].value_counts()
raw_count = raw_df['Image'].value_counts()

######

page_path_list = glob("C:/Users/alexh/Desktop/NII/newTest/tokaidochuhizakurige/*.jpg")
v3_data = pd.read_csv("C:/Users/alexh/Desktop/NII/newTest/predicted_res_v3_updated.csv", sep = ',',header = 0)

def display_result(input_data):
    page_id = list(set(input_data['Image']))
    for idx in page_id:
        image_path = 'C:/Users/alexh/Desktop/NII/newTest/' + idx
        output_path = 'C:/Users/alexh/Desktop/NII/newTest/display_result/' + idx
        prediction_page = input_data[input_data['Image'] == idx]
        coordinate_Val = list(zip(prediction_page.loc[:, 'x_center'], prediction_page.loc[:, 'y_center']))

        draw_line(coordinate_Val = coordinate_Val, image_path = image_path,
                  line_color = (0, 0, 255), output_path = output_path)

    return 'Completed'

display_result(v3_data)

v3 = predict_book(input_book_path, detect_line_version = 3, irregular_layout = True, only_return_performance = True)
v5_20 = predict_book(input_book_path, detect_line_version = 5, irregular_layout = True, only_return_performance = True)

def show_perf(input_result):
    print('coverage: ', input_result['coverage'].mean(axis = 0))
    print('edit norm: ', input_result['edit_norm'].mean(axis=0))
    print('bleu score: ', input_result['bleu_score'].mean(axis=0))

show_perf(v3)
show_perf(v5_10) # best
show_perf(v5_20)