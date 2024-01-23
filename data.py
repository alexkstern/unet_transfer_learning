from datasets_generations import save_cached_datasets_for_pelvicMR, save_cached_datasets_for_amos22


'''
This assumes the datasets follow the directory structure specified in the instructions.pdf,
Otherwise please change the paths in the config.py file accordingly.
'''

# when generating the datasets we used: random.seed(42)

# create cached datasets for the 1st dataset
save_cached_datasets_for_pelvicMR()

# create cached datasets for the 2nd dataset
save_cached_datasets_for_amos22()