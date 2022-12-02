# ====================================================
# CFG
# ====================================================
class FE:
    all_features = ['userID', 'assessmentItemID', 'testId', 'answerCode', 'Timestamp',
       'KnowledgeTag', 'answerCode_copy', 'before_ans_rate_testid_byuser',
       'down', 'user_mean', 'user_sum', 'test_mean', 'test_sum', 'tag_mean',
       'tag_sum', 'Item_mean', 'Item_sum', 'assessmentItemID_shift',
       'Item_mean_cum', 'weekday', 'assessment_main', 'assessment_mid',
       'assessment_sub', 'problem_time', 'assessment_main_mean',
       'assessment_main_count', 'assessment_main_problem_mean',
       'assessment_main_problem_count', 'assessment_mid_mean',
       'assessment_mid_count', 'assessment_mid_problem_mean',
       'assessment_mid_problem_count', 'assessment_sub_mean',
       'assessment_sub_count', 'assessment_sub_problem_mean',
       'assessment_sub_problem_count', 'year', 'month', 'day', 'hour',
       'minute', 'second', 'hour_mode']
    
    not_to_use = ['Timestamp', 'answerCode_copy', 'answerCode', 'assessment_mid_mean', 
				'year', 'day', 'weekday', 'minute', 'second']

    drop = ['Timestamp', 'answerCode_copy', 'assessment_mid_mean',  
				'year', 'day', 'weekday', 'minute', 'second']

    features = list(set(all_features)- set(not_to_use))


# ['Timestamp', 'answerCode_copy',  
# 				'year', 'day', 'weekday', 'minute', 'second', 'answerCode', 'assessment_mid_mean',
#             'Item_mean', 'assessment_main_count', 'assessment_sub_count', 'assessment_mid_count',
#             'assessment_main_mean', 'assessment_sub_mean', 'Item_mean_cum', 'Item_sum', 'test_mean'
#     ]