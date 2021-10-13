import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


from SatelCrawler import PatientList
data = PatientList(satel_directory="/home/aleis/Documents/SatelExport/SatelExportLocometreExport1.txt",
                   patient_info_csv="/home/aleis/Downloads/analyse marche tout - global data.csv",
                   path_to_analysis_folder="/210928/",
                   secondary_analysis=True,
                   path_to_csv_PMH= "/home/aleis/Downloads/analyse marche tout - PMHx simplified.csv",
                   path_to_MedsCSV= '/home/aleis/Downloads/analyse marche tout - Medication during locomotor testing.csv',
                   path_to_mmsCSV= '/home/aleis/Downloads/analyse marche tout - Cognitive tests.csv')



popdesc = data.analytics.loc[:, ["Nom", 'Nouveau groupes étiologiques',
                                 'Nouvelles classes étiologiques',
                                 'Etiologies (grands groupes)']].drop_duplicates()
index = pd.MultiIndex.from_frame(popdesc)
data.patient_info['matrice des valeurs numériques ou datations'].index = data.patient_info[
    'matrice des valeurs numériques ou datations'].index.str.lstrip(" ").str.replace("é", "e")
MedsDF = pd.DataFrame(data.MedsDuringTest.transpose().loc[:, index.get_level_values("Nom")].fillna(0).values,
                      columns=popdesc["Nom"], index=data.MedsDuringTest.columns.values)
PmhDF = pd.DataFrame(data.PMH.transpose().loc[:, index.get_level_values("Nom")].fillna(0).values,
                     columns=popdesc["Nom"], index=data.PMH.columns.values)
PatientInfoDF = pd.DataFrame(data.patient_info[
                                 'matrice des valeurs numériques ou datations'].loc[
                             index.get_level_values("Nom"), :].drop("Etiology", axis=1).transpose().values,
                             columns=popdesc["Nom"], index=data.patient_info[
        'matrice des valeurs numériques ou datations'].drop("Etiology", axis=1).columns)
PatientInfoDF.loc[["Time between exams"], :] = [pd.to_timedelta(i).days for i in
                                                PatientInfoDF.loc[["Time between exams"], :].values]
csv = pd.read_csv("/home/aleis/Downloads/analyse marche tout - Sex.csv")
csv.index = csv["NAME SURNAME"].str.lstrip(" ").str.replace("é", "e")
PatientInfoDF.loc["SEX", :] = csv.loc[index.get_level_values("Nom"), "SEX"].values
PatientInfoDF.loc["FDRCV", :] = data.patient_info['matrice des données exploitables avant - après PL'].loc[
    index.get_level_values("Nom"), "FdRCV"].values
# PatientInfoDF.loc["MMS before", "MMS after", :] = csv.loc[index.get_level_values("Nom"), "SEX"].values


# for _temp in ["Before LP", "After LP"]:
#     PatientInfoDF.loc[_temp, : ] = [i.timestamp() for i in PatientInfoDF.loc[_temp, :]]
# _tmp = pd.DataFrame()
# _tmp["orig"] = csv["DATE"]
# _tmp["copy"] = pd.to_datetime(csv["DATE"], dayfirst=True, errors="coerce")
# _tmp[_tmp["copy"].isna()] = np.nan
csv = pd.read_csv("/home/aleis/Downloads/analyse marche tout - Cog (5).csv")
csv.index = csv["NAME SURNAME"].str.lstrip(" ").str.replace("é", "e")
csv["DATE"] = pd.to_datetime(csv["DATE"], dayfirst=True, errors="coerce")
mmsDfunique = pd.DataFrame(columns=["MMS", "Date of MMS"])
mmsDF = pd.DataFrame(columns=["MMS BEFORE", "DATE MMS BEFORE", "MMS AFTER", "DATE MMS AFTER"])
for name in index.get_level_values("Nom"):
    _tmpdf = csv[csv.index == name][["RESULT", "DATE"]]
    print(_tmpdf)
    if len(_tmpdf.index) == 1:
        mmsDfunique.loc[name, :] = _tmpdf.loc[name, ["RESULT", "DATE"]].to_list()
    if len(_tmpdf.index) > 1:
        _tmp = [i == min(_tmpdf["DATE"]) for i in _tmpdf["DATE"]]
        print(_tmpdf[_tmp].loc[name, ["RESULT", "DATE"]].to_list())
        mmsDfunique.loc[name, :] = _tmpdf[_tmp].loc[name, ["RESULT", "DATE"]].to_list()
        mmsDF.loc[name, :] = _tmpdf[_tmp].loc[name, ["RESULT", "DATE"]].to_list() + \
                             _tmpdf[[i == max(_tmpdf["DATE"]) for i in _tmpdf["DATE"]]].loc[
                                 name, ["RESULT", "DATE"]].to_list()
mmsDF["DELTA MMS"] = pd.to_numeric(mmsDF["MMS AFTER"]) - pd.to_numeric(mmsDF["MMS BEFORE"])
mmsDF["DELTA TIME MMS"] = (mmsDF["DATE MMS AFTER"] - mmsDF["DATE MMS BEFORE"])
mmsDF["DELTA TIME MMS"] = [i.days for i in mmsDF["DELTA TIME MMS"]]

Compdiv = data.secondary_analysis_routine['Analyse comparée']['Comparaison: division (après:avant)']
Compdiff = data.secondary_analysis_routine['Analyse comparée']['Comparaison: soustraction (après - avant)']
Specdiv = data.secondary_analysis_routine['Analyse spectrale']['Comparaison TTF: division (après:avant)']
Specdiff = data.secondary_analysis_routine['Analyse spectrale']['Comparaison TTF: soustraction (après - avant)']
Specall = data.secondary_analysis_routine['Analyse spectrale']['Analyse tout échantillons TTF']
Specall_fil = pd.DataFrame(columns=Specall.columns)
for name in Specall["Nom"].unique():
    Specall_tab = Specall[Specall["Nom"] == name]
    head = Specall_tab.iloc[0, :6].values
    Specall_fil.loc[name, :] = head.tolist() + [Specall_tab[i].values.mean() for i in Specall.columns[6:]]

Compall = data.analytics
Compall_fil = pd.DataFrame(columns=data.analytics.columns)
for name in Compall["Nom"].unique():
    Compall_tab = Compall[Compall["Nom"] == name]
    head = Compall_tab.iloc[0, :6].values
    Compall_fil.loc[name, :] = head.tolist() + [Compall_tab[i].values.mean() for i in Compall.columns[6:]]

Compall_fil_clade = Compall_fil.iloc[:, :6]
Compall_fil.index = Compall_fil.Nom
Compall_fil_Stats = pd.DataFrame()
for clade_ in ['Nouveau groupes étiologiques', 'Nouvelles classes étiologiques', 'Etiologies (grands groupes)']:
    for select_ in Compall_fil_clade[clade_].unique():
        names = Compall_fil_clade.loc[Compall_fil_clade[clade_] == select_, "Nom"].values
        length = str(len(names))
        Compall_fil1 = Compall_fil.loc[names, :]
        Compall_fil1 = Compall_fil1.iloc[:, 6:]
        _tmp = "Regroupement: " + clade_ + ": " + select_
        Compall_fil_Stats[_tmp + " (Moyenne)"] = Compall_fil1.apply(np.mean)
        Compall_fil_Stats[_tmp + " (Std)"] = Compall_fil1.apply(np.std)
        Compall_fil_Stats[_tmp + " (Somme). n= " + length] = Compall_fil1.apply(np.sum)

Compdiv_clade = Compdiv.iloc[:, :6]
Compdiv.index = Compdiv.Nom
Compdiv_Stats = pd.DataFrame()
for clade_ in ['Nouveau groupes étiologiques', 'Nouvelles classes étiologiques', 'Etiologies (grands groupes)']:
    for select_ in Compdiv_clade[clade_].unique():
        names = Compdiv_clade.loc[Compdiv_clade[clade_] == select_, "Nom"].values
        length = str(len(names))
        Compdiv1 = Compdiv.loc[names, :]
        Compdiv1 = Compdiv1.iloc[:, 6:]
        _tmp = "Regroupement: " + clade_ + ": " + select_
        Compdiv_Stats[_tmp + " (Moyenne)"] = Compdiv1.apply(np.mean)
        Compdiv_Stats[_tmp + " (Std)"] = Compdiv1.apply(np.std)
        Compdiv_Stats[_tmp + " (Somme). n= " + length] = Compdiv1.apply(np.sum)

Compdiff_clade = Compdiff.iloc[:, :6]
Compdiff.index = Compdiff.Nom
Compdiff_Stats = pd.DataFrame()
for clade_ in ['Nouveau groupes étiologiques', 'Nouvelles classes étiologiques', 'Etiologies (grands groupes)']:
    for select_ in Compdiff_clade[clade_].unique():
        names = Compdiff_clade.loc[Compdiff_clade[clade_] == select_, "Nom"].values
        length = str(len(names))
        Compdiff1 = Compdiff.loc[names, :]
        Compdiff1 = Compdiff1.iloc[:, 6:]
        _tmp = "Regroupement: " + clade_ + ": " + select_
        Compdiff_Stats[_tmp + " (Moyenne)"] = Compdiff1.apply(np.mean)
        Compdiff_Stats[_tmp + " (Std)"] = Compdiff1.apply(np.std)
        Compdiff_Stats[_tmp + " (Somme). n= " + length] = Compdiff1.apply(np.sum)

Specdiv_clade = Specdiv.iloc[:, :6]
Specdiv.index = Specdiv.Nom
Specdiv_Stats = pd.DataFrame()
for clade_ in ['Nouveau groupes étiologiques', 'Nouvelles classes étiologiques', 'Etiologies (grands groupes)']:
    for select_ in Specdiv_clade[clade_].unique():
        names = Specdiv_clade.loc[Specdiv_clade[clade_] == select_, "Nom"].values
        length = str(len(names))
        Specdiv1 = Specdiv.loc[names, :]
        Specdiv1 = Specdiv1.iloc[:, 6:]
        _tmp = "Regroupement: " + clade_ + ": " + select_
        Specdiv_Stats[_tmp + " (Moyenne)"] = Specdiv1.apply(np.mean)
        Specdiv_Stats[_tmp + " (Std)"] = Specdiv1.apply(np.std)
        Specdiv_Stats[_tmp + " (Somme). n= " + length] = Specdiv1.apply(np.sum)

Specdiff_clade = Specdiff.iloc[:, :6]
Specdiff.index = Specdiff.Nom
Specdiff_Stats = pd.DataFrame()
for clade_ in ['Nouveau groupes étiologiques', 'Nouvelles classes étiologiques', 'Etiologies (grands groupes)']:
    for select_ in Specdiff_clade[clade_].unique():
        names = Specdiff_clade.loc[Specdiff_clade[clade_] == select_, "Nom"].values
        length = str(len(names))
        Specdiff1 = Specdiff.loc[names, :]
        Specdiff1 = Specdiff1.iloc[:, 6:]
        _tmp = "Regroupement: " + clade_ + ": " + select_
        Specdiff_Stats[_tmp + " (Moyenne)"] = Specdiff1.apply(np.mean)
        Specdiff_Stats[_tmp + " (Std)"] = Specdiff1.apply(np.std)
        Specdiff_Stats[_tmp + " (Somme). n= " + length] = Specdiff1.apply(np.sum)

Specall_fil_clade = Specall_fil.iloc[:, :6]
Specall_fil.index = Specall_fil.Nom
Specall_fil_Stats = pd.DataFrame()
for clade_ in ['Nouveau groupes étiologiques', 'Nouvelles classes étiologiques', 'Etiologies (grands groupes)']:
    for select_ in Specall_fil_clade[clade_].unique():
        names = Specall_fil_clade.loc[Specall_fil_clade[clade_] == select_, "Nom"].values
        length = str(len(names))
        Specall_fil1 = Specall_fil.loc[names, :]
        Specall_fil1 = Specall_fil1.iloc[:, 6:]
        _tmp = "Regroupement: " + clade_ + ": " + select_
        Specall_fil_Stats[_tmp + " (Moyenne)"] = Specall_fil1.apply(np.mean)
        Specall_fil_Stats[_tmp + " (Std)"] = Specall_fil1.apply(np.std)
        Specall_fil_Stats[_tmp + " (Somme). n= " + length] = Specall_fil1.apply(np.sum)

FinalAnalytics = pd.concat([Compall_fil_Stats.set_index("Comp. All " + Compall_fil_Stats.index.astype(str)),
                            Compdiv_Stats.set_index("Comp. Diff. " + Compdiv_Stats.index.astype(str)),
                            Compdiff_Stats.set_index("Comp. Div. " + Compdiff_Stats.index.astype(str)),
                            Specdiv_Stats.set_index("Spec. Div. " + Specdiv_Stats.index.astype(str)),
                            Specdiff_Stats.set_index("Spec. Diff. " + Specdiff_Stats.index.astype(str)),
                            Specall_fil_Stats.set_index("Spec. All " + Specall_fil_Stats.index.astype(str))],
                           axis=0)


beforeDF = data.secondary_analysis_routine['Analyse comparée']['DF avant'].set_index("Nom").iloc[:, 5:]
afterDF = data.secondary_analysis_routine['Analyse comparée']['DF_apres'].set_index("Nom").iloc[:, 5:]
afterDF_Stats = pd.DataFrame()
beforeDF_Stats = pd.DataFrame()
for clade_ in ['Nouveau groupes étiologiques', 'Nouvelles classes étiologiques', 'Etiologies (grands groupes)']:
    for select_ in popdesc[clade_].unique():
        names = popdesc.loc[popdesc[clade_] == select_, "Nom"].values
        length = str(len(names))
        beforeDF_Stats["Regroupement: " + clade_ + ": "+ select_+" (Moyenne)"] = beforeDF.loc[names,:].fillna(0).apply(np.nanmean)
        beforeDF_Stats["Regroupement: " + clade_ + ": "+ select_+" (Std)"] = beforeDF.loc[names,:].fillna(0).apply(np.std)
        afterDF_Stats["Regroupement: " + clade_ + ": "+ select_+" (Moyenne)"] = afterDF.loc[names,:].fillna(0).apply(np.nanmean)
        afterDF_Stats["Regroupement: " + clade_ + ": "+ select_+" (Std)"] = afterDF.loc[names,:].fillna(0).apply(np.std)

names_NoNPH, names_NPH = [popdesc.loc[popdesc['Etiologies (grands groupes)'] == select_, "Nom"].values
                          for select_ in popdesc['Etiologies (grands groupes)'].unique()]
for tab_, tab0_, namesall in zip([beforeDF_Stats, afterDF_Stats, FinalAnalytics],
                       [afterDF.transpose(), beforeDF.transpose(), MotherOfDf.fillna(np.nan).transpose()], ["Tests avant PL", "Tests après PL", "Tests tous"]):
    for item_ in tab_.index:
        try:
            tab_.loc[item_, "Regroupement: " + 'Etiologies (grands groupes)' + ": "+ " (t-test)"] = stats.ttest_ind(
                tab0_.loc[item_, names_NPH].values, tab0_.loc[item_, names_NoNPH].values,
                equal_var=True, nan_policy='omit')[0]
            tab_.loc[item_, ["Regroupement: " + 'Etiologies (grands groupes)' + ": "+ " (p-value)"]] = stats.ttest_ind(
                tab0_.loc[item_, names_NPH].values, tab0_.loc[item_, names_NoNPH].values,
                equal_var=True, nan_policy='omit')[1]
        except TypeError:
            print(item_)

MotherOfDf = pd.concat(
    [Compall_fil.iloc[:, 6:].add_prefix("Comp. All "), Compdiv.iloc[:, 6:].add_prefix("Comp. Diff. "),
     Compdiff.iloc[:, 6:].add_prefix("Comp. Div. "), Specdiv.iloc[:, 6:].add_prefix("Spec. Div. "),
     Specdiff.iloc[:, 6:].add_prefix("Spec. Diff. "), Specall_fil.iloc[:, 6:].add_prefix("Spec. All ")], axis=1)


MedsDF_Stats = pd.DataFrame()
PmhDF_Stats = pd.DataFrame()
PatientInfoDF_Stats = pd.DataFrame()
for clade_ in ['Nouveau groupes étiologiques', 'Nouvelles classes étiologiques', 'Etiologies (grands groupes)']:
    for select_ in popdesc[clade_].unique():
        names = popdesc.loc[popdesc[clade_] == select_, "Nom"].values
        length = str(len(names))
        MedsDF_Stats["Regroupement: " + clade_ + ": "+ select_+" (Moyenne)"] = MedsDF.loc[:,names].fillna(0).transpose().apply(np.nanmean)
        MedsDF_Stats["Regroupement: " + clade_ + ": "+ select_+" (Std)"] = MedsDF.loc[:,names].fillna(0).transpose().apply(np.std)
        MedsDF_Stats["Regroupement: " + clade_ + ": "+ select_+" (Somme). n= " + length] = MedsDF.loc[:,names].fillna(0).transpose().apply(np.sum)
        PmhDF_Stats["Regroupement: " + clade_ + ": "+ select_+" (Moyenne)"] = PmhDF.loc[:,names].transpose().apply(np.nanmean)
        PmhDF_Stats["Regroupement: " + clade_ + ": "+ select_+" (Std)"] = PmhDF.loc[:,names].transpose().apply(np.std)
        PmhDF_Stats["Regroupement: " + clade_ + ": "+ select_+" (Somme). n= " + length] = PmhDF.loc[:,names].transpose().apply(np.sum)
        PatientInfoDF_Stats["Regroupement: " + clade_ + ": "+ select_+" (Moyenne)"] = PatientInfoDF.loc["Time between exams":,names].fillna(0).transpose().apply(np.nanmean)
        PatientInfoDF_Stats["Regroupement: " + clade_ + ": "+ select_+" (Std)"] = PatientInfoDF.loc["Time between exams":,names].fillna(0).transpose().apply(np.std)
        # print(MedsDF.loc[:,names].fillna(0).apply(np.std))

names_NoNPH, names_NPH = [popdesc.loc[popdesc['Etiologies (grands groupes)'] == select_, "Nom"].values
                          for select_ in popdesc['Etiologies (grands groupes)'].unique()]
for tab_, tab0_, namesall in zip([MedsDF_Stats, PmhDF_Stats, PatientInfoDF_Stats],
                       [MedsDF, PmhDF, PatientInfoDF.loc["Time between exams":,:].fillna(np.nan)], ["Meds pendant le test", "ATCDs", "Resumé carac patients"]):
    for item_ in tab_.index:
        try:
            tab_.loc[item_, "Regroupement: " + 'Etiologies (grands groupes)' + ": "+ " (t-test)"] = stats.ttest_ind(
                tab0_.loc[item_, names_NPH].values, tab0_.loc[item_, names_NoNPH].values,
                equal_var=True, nan_policy='omit')[0]
            tab_.loc[item_, ["Regroupement: " + 'Etiologies (grands groupes)' + ": "+ " (p-value)"]] = stats.ttest_ind(
                tab0_.loc[item_, names_NPH].values, tab0_.loc[item_, names_NoNPH].values,
                equal_var=True, nan_policy='omit')[1]
        except TypeError:
            print(item_)


FinalAnalytics.to_excel("/home/aleis/Documents/1.Workspace/1.Gait_analysis/210928/WholeAnalytics.xlsx")
beforeDF_Stats.to_excel("/home/aleis/Documents/1.Workspace/1.Gait_analysis/210928/beforeAnalytics.xlsx")
afterDF_Stats.to_excel("/home/aleis/Documents/1.Workspace/1.Gait_analysis/210928/afterAnalytics.xlsx")
