import argparse
import pandas as pd
import numpy as np
import os


# python3  /home/mw/Repos/itc-main-repo/Ming/pytorch/split_sample.py
# -p
# /home/mw/Analyses/deeplearning_UKB_mri/2019-08-11.DataSelection.dta.visit1full.Rphenotypes.merged_CMRsusbet_DLlabels.tsv
# -il /home/mw/Analyses/deeplearning_UKB_mri/all_id_img_path.txt
# -v
# sex
# -i
# app12010

def read_phenotype(phenotype_file, target_variables, eid_variable, save=True, outname=None, dropNA=True):
    '''

    :param phenotype_file: string, tab delimited text file storing the phenotypes with header
    :param target_variables: list, list of target variables
    :param eid_variable: string , the name of column of the relevant eid
    :return: pd.dataframe, target variables and eid
    '''

    df_phen = pd.read_csv(phenotype_file, sep='\t', header=0,)
    if dropNA:
        df_phen = df_phen.dropna()
    print(list(df_phen))
    cols = target_variables.copy()
    cols.insert(0, eid_variable)
    # cols.append(eid_variable)
    print(target_variables, eid_variable, cols)
    # print(df_phen[df_phen[eid] == 1023067])

    df_sub = df_phen[cols]
    # print(df_sub[df_sub[eid] == 1023067])
    if save is True and outname is not None:
        df_sub.to_csv(outname, sep='\t', na_rep='', float_format=None, columns=None, header=False, index=False,
                      index_label=None)
    return df_sub


def random_binary_split_sample(df, fraction, seed, save=True, outname1=None, outname2=None):
    # pandas version
    df_frac1 = df.sample(frac=fraction, random_state=seed)
    ## numpy version
    ## Chose how many index include for random selection
    # chosen_idx = np.random.choice(4, replace=True, size=6)
    # df2 = df.iloc[chosen_idx]

    col_name = list(df.columns.values)
    # get the rows not in df_fac1
    df_frac2 = df.merge(df_frac1, on=col_name, how='left', indicator=True)
    df_frac2 = df_frac2[df_frac2['_merge'] == 'left_only']
    df_frac2 = df_frac2[col_name]
    # print(df, df_frac1, df_frac2)

    if save is True and outname1 is not None and outname2 is not None:
        df_frac1.to_csv(outname1, sep='\t', na_rep='', float_format=None, columns=None, header=False, index=False,
                        index_label=None)
        df_frac2.to_csv(outname2, sep='\t', na_rep='', float_format=None, columns=None, header=False, index=False,
                        index_label=None)
    return df_frac1, df_frac2


if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-p', '--phenotype', required=True,
                    help='phenotype file / residual file')
    ap.add_argument("-il", "--imgLst", required=False,default=None,
                    help="file listing filename of all images")
    ap.add_argument('-v', '--variables', required=True,
                    help='target variables,comma separated')
    ap.add_argument('-i', '--eid', required=True,
                    help='unique identifier that connect the phenotype and images')
    # ap.add_argument('-os', '--outnameSubset', required=False,default='subset',
    #                 help='name of output file with selected variables')
    # ap.add_argument('-ot', '--outnameTrain', required=False,default='.train',
    #                 help='name of output file with training set')
    # ap.add_argument('-ov', '--outnameVal', required=False,default='.val',
    #                 help='name of output file with validation set')
    # ap.add_argument('-oe', '--outnameExtTest', required=False,default='.test',
    #                 help='name of output file with test set')
    ap.add_argument('-f', '--fractions', required=False, default='0.7,0.2,0.1',
                    help='fractions of trainging , validation and test set')
    ap.add_argument('-r', '--randomSeed', required=False, default=255,
                    help='custom seed')
    ap.add_argument('-na', '--na', required=False, default=False,
                    help='drop NA')

    args = vars(ap.parse_args())
    input_phenotype = args['phenotype']
    img_list = args['imgLst']
    var_list = list(args['variables'].split(','))
    eid = args['eid']
    na=eval(args['na'])

    print("phenotype file: {}".format(input_phenotype))
    print("image list file: {}".format(img_list))
    print("selected variable(s): {}".format(var_list))
    print("name of the eid column:{}".format(eid))

    try:
        fracs = list(args['fractions'].split(','))
    except:
        fracs = list(args['fractions'])
    seed = args['randomSeed']
    # these options seem unnecessary, drop for now - May2020
    # try:
    #     out_subset = args['outnameSubset']
    #     out_train = args['outnameTrain']
    #     out_val = args['outnameVal']
    #     out_test = args['outnameTest']
    # except:
    wd = os.path.dirname(input_phenotype)
    print(wd)

    pheno_basename = os.path.basename(input_phenotype)
    out_sub = '_'.join(var_list)
    print(out_sub)
    out_prefix = pheno_basename + "_" + out_sub
    out_subset = os.path.join(wd, out_prefix + '.tsv')
    out_train = os.path.join(wd, out_prefix + '.train')
    out_val = os.path.join(wd, out_prefix + '.val')
    out_test = os.path.join(wd, out_prefix + '.test')
    # out_temp=os.path.join(wd, out_sub, 'temp.csv')

    # first extract variables
    df_selected = read_phenotype(input_phenotype, var_list, eid, save=True, outname=out_subset,dropNA=na)

    # split the data
    if (float(fracs[2]) + float(fracs[1]) + float(fracs[0])) != 1:
        print("fraction in total :{} !=1 ".format((float(fracs[2]) + float(fracs[1]) + float(fracs[0]))))
        print("please check the fractions of the splits...................")

    frac_test = float(fracs[2])
    frac_trainval = 1 - frac_test
    frac_train = float(fracs[0]) / (float(fracs[0]) + float(fracs[1]))
    df_trainval, df_test = random_binary_split_sample(df_selected, frac_trainval, seed, False)
    df_train, df_val = random_binary_split_sample(df_trainval, frac_train, seed, False)

    print("training/validation/test set n= {0} / {1} / {2}\n Write to files".format(len(df_train), len(df_val),
                                                                                    len(df_test)))

    df_train.to_csv(out_train, sep='\t', na_rep='', float_format=None, columns=None, header=False, index=False,
                    index_label=None)
    df_val.to_csv(out_val, sep='\t', na_rep='', float_format=None, columns=None, header=False, index=False,
                  index_label=None)
    df_test.to_csv(out_test, sep='\t', na_rep='', float_format=None, columns=None, header=False, index=False,
                   index_label=None)
    if img_list is not None:
        print('parse the image path file based on the current split')
        id_train = df_train[eid]
        id_val = df_val[eid]
        id_test = df_test[eid]
        imgLst_basename = os.path.basename(img_list)
        out_prefix = imgLst_basename + "_" + out_sub

        out_train = os.path.join(wd, out_prefix + '.train')
        out_val = os.path.join(wd, out_prefix + '.val')
        out_test = os.path.join(wd, out_prefix + '.test')
        df_imgLst = pd.read_csv(img_list, sep='\t', header=None, names=[eid, "imgPath"])
        df_imgLst_train = df_imgLst[df_imgLst[eid].isin(id_train)]
        df_imgLst_val = df_imgLst[df_imgLst[eid].isin(id_val)]
        df_imgLst_test = df_imgLst[df_imgLst[eid].isin(id_test)]
        df_imgLst_train.to_csv(out_train, sep='\t', na_rep='', float_format=None, columns=None, header=False, index=False,
                               index_label=None)
        df_imgLst_val.to_csv(out_val, sep='\t', na_rep='', float_format=None, columns=None, header=False, index=False,
                             index_label=None)
        df_imgLst_test.to_csv(out_test, sep='\t', na_rep='', float_format=None, columns=None, header=False, index=False,
                              index_label=None)

    print("Finish")

    # if len(var_list) == 1:
    #     # if there is one target variable , create respective id lists
    #     # this serves as a temporary solution with no capability to handle multiple variables at once
    #     for group in np.unique(df_selected[var_list[0]].values):
    #         df_subgroup_train = df_train[df_train[var_list[0]] == group]
    #         df_subgroup_val = df_val[df_val[var_list[0]] == group]
    #         df_subgroup_test = df_test[df_test[var_list[0]] == group]
    #         id_train = df_subgroup_train[eid]
    #         id_val = df_subgroup_val[eid]
    #         id_test = df_subgroup_test[eid]
    #         out_train = os.path.join(wd, out_sub + '_train_{}'.format(group))
    #         out_val = os.path.join(wd, out_sub + '_val_{}'.format(group))
    #         out_test = os.path.join(wd, out_sub + '_test_{}'.format(group))
    #         id_train.to_csv(out_train, sep='\t', na_rep='', float_format=None, columns=None, header=False, index=False,
    #                         index_label=None)
    #
    #         id_val.to_csv(out_val, sep='\t', na_rep='', float_format=None, columns=None, header=False, index=False,
    #                       index_label=None)
    #
    #         id_test.to_csv(out_test, sep='\t', na_rep='', float_format=None, columns=None, header=False, index=False,
    #                        index_label=None)
    # else:
    #     id_train = df_train[eid]
    #     id_val = df_val[eid]
    #     id_test = df_test[eid]
    #     id_train.to_csv(out_train, sep='\t', na_rep='', float_format=None, columns=None, header=False, index=False,
    #                     index_label=None)
    #
    #     id_val.to_csv(out_val, sep='\t', na_rep='', float_format=None, columns=None, header=False, index=False,
    #                   index_label=None)
    #
    #     id_test.to_csv(out_test, sep='\t', na_rep='', float_format=None, columns=None, header=False, index=False,
    #                    index_label=None)
