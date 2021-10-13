from spectrum import Spectrogram, Periodogram, readwav, aryule
import os
import re
import datetime
from datetime import datetime
import numpy as np
import unidecode as unidecode
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import peak_prominences
from scipy.signal import peak_widths
from scipy.fft import  fft, fftfreq
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
pd.set_option('use_inf_as_na', True)
import mne

plt.ioff()

def nearest2(items, pivot):
    delta = [i - pivot for i in items]
    return [min([abs(j) for j in delta if j.days<=0]), pivot-min([abs(j) for j in delta if j.days<=0]) , min([abs(j) for j in delta if j.days>=0]), pivot + min([abs(j) for j in delta if j.days>=0])]

class PatientList:
    print("Patient data acquisition...")
    def __init__(self, satel_directory, patient_info_csv=None, satel_link=None,
                 path_to_analysis_folder=None, secondary_analysis=False, path_to_csv_PMH = None,
                 path_to_MedsCSV= None, path_to_mmsCSV = None):
        self.satel_directory = satel_directory
        self.satel_link = satel_link
        self.path_to_mmsCSV = path_to_mmsCSV
        self.patient_info_csv = patient_info_csv
        self.path_to_analysis_folder = path_to_analysis_folder
        self.gait_data = self.parser_satel()
        if patient_info_csv is not None:
            self.patient_info = self.ClinicalDataParser(path_to_analysis_folder)
            if path_to_csv_PMH is not None:
                self.path_to_csv_PMH = path_to_csv_PMH
                self.PMH = self.CardiovascularRisk()
            self.path_to_MedsCSV = path_to_MedsCSV
            if self.path_to_MedsCSV is not None:
                self.MedsDuringTest = self.MedsDuringTest()
            self.satel_link = self.SatelLink()
            self.analytics = self.BaseAnalytics()
            if self.path_to_mmsCSV is not None:

                self.mms = self.MMSTest()
        if secondary_analysis is not False:
            print("Ordered secondary analytics: placed on queue...")
            self.secondary_analysis = self.secondary_analysis_routine()

    def parser_satel(self):
        print("Parsing Satel Locometer output...")
        with open(self.satel_directory, 'r', encoding="ISO-8859-1") as file:
            i = 0
            sep = [0]
            file_lines = file.readlines()
            for line in file_lines:
                if line == '\n':
                    sep.append(i)
                i = i + 1
            head_comments = file_lines[0].strip("\n").split(";")
            df_patients = []
            loco_patients = dict()
            i=0
            for _tmp in range(len(sep)-1):
                first_header = file_lines[sep[_tmp]+1].strip("\n").split(";")
                df_patients.append(first_header)
                range_rows = sep[_tmp+1] - sep[_tmp]-5
                name = first_header[0].split("-")[-1]
                loco_df = pd.read_table(self.satel_directory, header=2,
                                    sep=";", encoding="ISO-8859-1", skiprows=sep[_tmp], nrows=range_rows)
                if type(loco_df.iloc[0, 0]) == str:
                    loco_df = loco_df.reset_index()
                    loco_df.columns = ['Temps ms', 'pied gauche', 'pied droit', "Synchro Video"]
                    loco_df = loco_df.iloc[1:,]
                SatelKeyID = name + " "+ first_header[1]+ ": "+ first_header[8] + " " + first_header[9]
                loco_patients[SatelKeyID] = loco_df
                i=i+1
            df_patients = pd.DataFrame(df_patients, columns=head_comments)
            df_patients[['Protocole', 'Nom']] = df_patients.NomPatient.str.split("-", expand=True)
            cols = df_patients.columns.tolist()
            cols = cols[-2:] + cols[:-2]
            df_patients = df_patients[cols]
            return({"resumé valeurs": df_patients, "locometre": loco_patients})
    def ClinicalDataParser(self, path_to_analysis_folder):
        print("Joining data and extracting indices that it might be related to LP dates...")
        self.path_to_analysis_folder= path_to_analysis_folder
        relevant_patient = pd.read_csv(self.patient_info_csv)
        list_clinical_data_name = relevant_patient["NAME SURNAME"].tolist()
        i = -1
        list_names_relevant = []
        list_index = []
        columns = self.gait_data['resumé valeurs'].columns.to_list()
        df_loco_value = pd.DataFrame(columns=["index locometry", "clinical_names"])
        for last_name, first_name, date, hours in list(
                zip(self.gait_data['resumé valeurs'][columns[1]], self.gait_data['resumé valeurs'][columns[3]],
                    self.gait_data['resumé valeurs'][columns[10]], self.gait_data['resumé valeurs'][columns[11]])):
            i = i + 1
            for clinical_data_name in list_clinical_data_name:
                if type(last_name) != str:
                    continue
                if type(first_name) != str:
                    continue
                if re.search(last_name.lower().strip(" "), clinical_data_name.lower()) and \
                        re.search(first_name.replace("é", "e").lower().strip(" ")[0:3],
                                  clinical_data_name.replace("é", "e").lower()):
                    list_names_relevant.append(clinical_data_name)
                    list_index.append(i)
                    # dict_loco_values[last_name + " " + first_name + ": " + date + " " + hours] = clinical_data_name
                    df_loco_value.loc[last_name + " " + first_name + ": " + date + " " + hours] = [i,
                                                                                                   clinical_data_name]
        list_index = list(set(list_index))
        loco_filtered = dict()
        i = -1
        for key in self.gait_data['locometre'].keys():
            i = i + 1
            if i in list_index:
                loco_filtered[key] = self.gait_data['locometre'][key]
        df_exploded = df_loco_value.merge(self.gait_data['resumé valeurs'], left_on='index locometry', right_index=True,
                                          how='left')
        df_exploded = df_exploded.merge(relevant_patient, left_on='clinical_names', right_on='NAME SURNAME', how='left')

        df_exploded['Nom Prenom'] = df_exploded[["Nom", 'PrenomPatient']].agg(' '.join, axis=1)
        dates_dict_by_patient = {}
        # TODO find all special cases and rule them out programatically (probably using date spans/datetime
        # select those who are already labeled
        regex = "ava"
        series = df_exploded["CommentaireEnreg"]
        df1 = series.str.contains(regex, case=False, flags=re.IGNORECASE, regex=True)
        df1 = df_exploded[df1]
        df1 = df1[df1["DateEnreg"] != "09/10/2018"] #special case
        df1 = df1[["Nom Prenom", "DateEnreg"]].drop_duplicates()

        regex = "apr"
        series = df_exploded["CommentaireEnreg"]
        df1b = series.str.contains(regex, case=False, flags=re.IGNORECASE, regex=True)
        df1b = df_exploded[df1b]
        df1b = df1b[["Nom Prenom", "DateEnreg"]].drop_duplicates()
        df1c = df_exploded[df_exploded["DateEnreg"] == "10/03/2008"][["Nom Prenom", "DateEnreg"]] #special case

        # select double dates wink wink
        df2 = df_exploded[["Nom Prenom", "DateEnreg"]]
        serie_fil = []
        name_list = []
        for ID in set(df2['Nom Prenom'].to_list()):
            result = len(set(df2.loc[df2["Nom Prenom"] == ID]["DateEnreg"].to_list())) == 2
            serie_fil.append(result)
            if result == True:
                name_list.append(ID)
            (set(df2.loc[df2["Nom Prenom"] == ID]["DateEnreg"].to_list()))
        boolean_series = df2["Nom Prenom"].isin(name_list)
        df2 = df2[boolean_series][["Nom Prenom", "DateEnreg"]]

        df3 = df_exploded["Nom Prenom"] == 'LEGRIX Christian'
        df3 = df_exploded[df3]
        df3 = df3[df3["DateEnreg"] != "23/11/2004"][["Nom Prenom", "DateEnreg"]]

        df_filtered_final = pd.concat([df1, df1b, df1c, df2, df3], ignore_index=True).drop_duplicates()

        df_filtered_final["filtered results"] = df_filtered_final[["Nom Prenom", "DateEnreg"]].agg(' '.join, axis=1)
        df_exploded["filtered results"] = df_exploded[["Nom", 'PrenomPatient', 'DateEnreg']].agg(' '.join, axis=1)
        boolean_series = df_exploded["filtered results"].isin(df_filtered_final["filtered results"].to_list())
        final_filtered_df = df_exploded[boolean_series]
        final_filtered_df = final_filtered_df[
            final_filtered_df[
                "DateEnreg"] != "22/09/2020|23/07/2007|26/07/2007|22/03/2010|30/03/2017|31/03/2017|17/06/2013"]
        reject = ["17/11/2009", "22/09/2020", "23/07/2007", "26/07/2007", "22/03/2010", "30/03/2017", "31/03/2017",
                  "17/06/2013"]
        inverse_boolean_series = ~final_filtered_df.DateEnreg.isin(reject)
        final_filtered_df = final_filtered_df[inverse_boolean_series]
        # TODO very high rate of special cases... guess if someone is looking at this in the future (lulz) he has to work on that
        # TODO export final_filtered_df
        # descriptive statistics
        data_statistics_LP = pd.DataFrame(
            columns=["Patient", "Before LP", "After LP", "Time between exams", "DOB", "Age", "Weight", "Height",
                     "Etiology"])
        for DOB, patient, weight, height, etiology in zip(final_filtered_df["DOB"].to_list(),
                                                          final_filtered_df["Nom Prenom"].to_list(),
                                                          pd.to_numeric(final_filtered_df['poids'], errors='coerce'),
                                                          pd.to_numeric(final_filtered_df['taille'], errors='coerce'),
                                                          final_filtered_df['etiology'].to_list()):
            list_dates_exp = [tmp.strip("`") for tmp in
                              final_filtered_df[final_filtered_df["Nom Prenom"] == patient]['DateEnreg']]
            list_dates_exp = pd.to_datetime(list_dates_exp, dayfirst=True).unique().tolist()
            age = min(list_dates_exp) - datetime.strptime(DOB, "%d/%m/%Y")
            age = int(age.days / 365.25)
            data_statistics_LP.loc[patient] = [patient, min(list_dates_exp), max(list_dates_exp),
                                               max(list_dates_exp) - min(list_dates_exp),
                                               DOB, age, weight, height, etiology]
        # TODO: separate the datetime/timestamp usefullness... probably makes more sense to have everything datetime
        data_statistics_LP_math = pd.DataFrame(
            columns=["Before LP", "After LP", "Time between exams", "Age", "Weight", "Height", "Etiology"])
        for DOB, patient, weight, height, etiology in zip(final_filtered_df["DOB"].to_list(),
                                          final_filtered_df["Nom Prenom"].to_list(),
                                          pd.to_numeric(final_filtered_df['poids'], errors='coerce'),
                                          pd.to_numeric(final_filtered_df['taille'], errors='coerce'),
                                          final_filtered_df['etiology'].to_list()):
            list_dates_exp = [tmp.strip("`") for tmp in
                              final_filtered_df[final_filtered_df["Nom Prenom"] == patient]['DateEnreg']]
            list_dates_exp = pd.to_datetime(list_dates_exp, dayfirst=True).unique().tolist()
            age = min(list_dates_exp) - datetime.strptime(DOB, "%d/%m/%Y")
            age = int(age.days / 365.25)
            data_statistics_LP_math.loc[patient] = [min(list_dates_exp),
                                                    max(list_dates_exp),
                                                    (max(list_dates_exp) - min(list_dates_exp)),
                                                    age, weight, height, etiology]
        # TODO export a math dataframe: probz data_statistics_LP_math
        stats_LP = pd.DataFrame(columns=data_statistics_LP_math.columns)
        stats_LP.loc["mean"] = data_statistics_LP_math.mean(numeric_only=True, skipna=True)
        stats_LP.loc["median"] = data_statistics_LP_math.median(numeric_only=True)
        stats_LP.loc["min"] = data_statistics_LP_math.min(numeric_only=True)
        stats_LP.loc["max"] = data_statistics_LP_math.max(numeric_only=True)
        stats_LP.loc["DevStd"] = data_statistics_LP_math.mad()
        stats_LP_format = stats_LP[stats_LP.columns]
        for column in stats_LP.columns[[2]]:
            stats_LP_format[column] = pd.to_timedelta(stats_LP[column])
        stats_LP.loc['DevStd', ["Before LP", "After LP", "Time between exams"]] = [i / (360 * 60 * 24) for i in
                                                                                   stats_LP.loc['DevStd', ["Before LP",
                                                                                                           "After LP",
                                                                                                           "Time between exams"]]]
        stats_LP_format.loc["DevStd"] = stats_LP.loc["DevStd"]
        dates_statistics = pd.DataFrame(columns=["minimum", "maximum", "delta", "n studies"])
        for key, value in dates_dict_by_patient.items():
            value = [datetime.strptime(datetime_string, "%d/%m/%Y").timestamp() for datetime_string in value]
            d0 = datetime.fromtimestamp(max(value)) - datetime.fromtimestamp(min(value))
            dates_statistics.loc[key] = [datetime.fromtimestamp(min(value)), datetime.fromtimestamp(max(value)),
                                         d0.days, len(value)]

        grouped_satel = final_filtered_df[["etiology", "clinical_names"]]
        grouped_satel = pd.concat(
            [grouped_satel, (pd.to_datetime(final_filtered_df["DateEnreg"], infer_datetime_format=True))], axis=1,
            ignore_index=False)
        grouped_satel_duplicates_dropped = grouped_satel.drop_duplicates().groupby( ['etiology', "clinical_names"])
        grouped_satel = grouped_satel.groupby( ["etiology", "clinical_names"])

        list_kept_for_two = pd.DataFrame()
        list_dropped_over_two = pd.DataFrame()
        list_dropped_for_one = pd.DataFrame()
        for name, group in grouped_satel_duplicates_dropped:
            if len(group) == 1:
                list_dropped_for_one[group["clinical_names"].iloc[0]] = [group["DateEnreg"].iloc[0],
                                                                         group["clinical_names"], group["etiology"]]
            if len(group) == 2:
                list_kept_for_two[group["clinical_names"].iloc[0]] = [group["DateEnreg"].min(),
                                                                      group["DateEnreg"].max(), (
                                                                                  group["DateEnreg"].max() - group[
                                                                              "DateEnreg"].min()),
                                                                      group["etiology"].iloc[0]].copy()
            if len(group) > 2:
                list_dropped_over_two[group["clinical_names"].iloc[0]] = [group["DateEnreg"].min(),
                                                                          group["DateEnreg"].max(), (
                                                                                      group["DateEnreg"].max() - group[
                                                                                  "DateEnreg"].min()),
                                                                          group["DateEnreg"].unique(),
                                                                          group["etiology"].iloc[0]]

        list_kept_for_two = list_kept_for_two.transpose()
        list_kept_for_two = list_kept_for_two.copy()
        list_kept_for_two = list_kept_for_two.rename(columns={0: "First", 1: "Last", 2: "Delta", 3: "Etiology"})
        list_kept_for_two["Name"] = list_kept_for_two.index
        listeis = list_kept_for_two["Etiology"].str.replace(r"HTIC.+", "HTIC")
        listeis = listeis.str.replace(r"Hydroc.+", "Hydrocéphalie")
        listeis = listeis.str.replace(r".+park.+", "Syndrome parkinsonien")
        listeis = listeis.str.replace(r"syndrome depre.+", "Dépression")
        listeis = listeis.str.replace(r".+neuropathi.+|.+ciatique.+|.+lombaire.+|myel.+", "Atteinte périphérique")
        listeis = listeis.str.replace(r".+alzhei.+", "Alzheimer")
        listeis = listeis.str.replace(
            r"Cryptococcose.+|paralysie supramucleaire progressive|demence a corps de lewy|.+amyloid.+|.+radi.+|gliome",
            "Autre")
        listeis = listeis.str.replace(
            r".+asepti.+|maladie de steinert|paraparesie spastique|sclerose en plaques|traumatisme bifrontal|iatrogenie",
            "Autre")
        listeis = listeis.str.replace(r"demence mixte", "Démence mixte")
        listeis = listeis.str.replace(r"demence vasculaire", "Démence vasculaire")

        listeis_HPN = listeis.str.replace(
            r'HTIC|Hydrocéphalie|Atteinte périphérique|Démence mixte|Démence vasculaire|Alzheimer|Dépression|Syndrome parkinsonien',
            "Autre")
        listeis_HPN_grouped = listeis_HPN.str.replace(r'HPNi|HPNs', "HPN")
        # TODO export list_kept_for_two
        list_kept_for_two['Nouveau groupes etiologiques'] = listeis
        list_kept_for_two['Nouvelles classes etiologiques'] = listeis_HPN
        list_kept_for_two['Etiologies (grands groupes)'] = listeis_HPN_grouped
        if path_to_analysis_folder is not None:
            grouped_satel_duplicates_dropped.describe().to_excel(
                path_to_analysis_folder + "SatelExportLocometre_Export1_summary_Stats_pop_by_dates.xlsx")
            list_kept_for_two.groupby( "Etiology").describe().to_excel(
                path_to_analysis_folder + "SatelExportLocometre_Export1_summary_Stats_pop_Dates_by_Etiology.xlsx")
            df_exploded.to_excel(path_to_analysis_folder + "/SatelExportLocometreExport1_exploded.xlsx")
            data_statistics_LP_math.groupby( "Etiology", as_index=True).describe().to_excel(
                path_to_analysis_folder + "SatelExportLocometre_Export1_summary_Stats_pop_by_etiology.xlsx")
            final_filtered_df[["etiology", "DateEnreg", "clinical_names"]].groupby(
                ["etiology", "clinical_names"]).describe().to_excel(
                path_to_analysis_folder + "SatelExportLocometre_Export1_summary_Stats_pop_by_number_of_dates.xlsx")
        return({'matrices des valeurs explosée': df_exploded, 'matrice des valeurs filtrée': final_filtered_df,
                'matrice des valeurs numériques ou datations': data_statistics_LP_math,
                'matrice des données exploitables avant - après PL': list_kept_for_two})

    def SatelLink(self):
        print("Linking Locometer and clinical data")
        Satel_analytics = pd.DataFrame(columns=["Nom", "Avant/Après", "Satel Key ID", 'Nouveau groupes étiologiques',
                                                'Nouvelles classes étiologiques',
                                                'Etiologies (grands groupes)'])
        for i in range(len(self.patient_info['matrice des données exploitables avant - après PL'].index)):
            name = self.patient_info['matrice des données exploitables avant - après PL'].index[i]
            item1 = self.patient_info['matrice des données exploitables avant - après PL']["First"][i].strftime(
                "%d/%m/%Y")
            item2 = self.patient_info['matrice des données exploitables avant - après PL']["Last"][i].strftime(
                "%d/%m/%Y")
            # self.patient_info['matrice des données exploitables avant - après PL']["Last"][i]
            keys_1 = [i_ for i_ in list(self.gait_data["locometre"].keys()) if
                      item1 in i_ and unidecode.unidecode(name) in unidecode.unidecode(i_)]
            keys_2 = [i_ for i_ in list(self.gait_data["locometre"].keys()) if
                      item2 in i_ and unidecode.unidecode(name) in unidecode.unidecode(i_)]
            etio = list(self.patient_info['matrice des données exploitables avant - après PL'][[
                'Nouveau groupes etiologiques',
                'Nouvelles classes etiologiques',
                'Etiologies (grands groupes)']].iloc[i])
            df_1 = pd.DataFrame(np.array([[name, "Avant", etio[0], etio[1], etio[2]]] * len(keys_1)),
                                columns=["Nom", "Avant/Après",
                                         'Nouveau groupes étiologiques',
                                         'Nouvelles classes étiologiques',
                                         'Etiologies (grands groupes)'])
            df_2 = pd.DataFrame(np.array([[name, "Après", etio[0], etio[1], etio[2]]] * len(keys_2)),
                                columns=["Nom", "Avant/Après",
                                         'Nouveau groupes étiologiques',
                                         'Nouvelles classes étiologiques',
                                         'Etiologies (grands groupes)'])
            df_1["Satel Key ID"] = keys_1
            df_2["Satel Key ID"] = keys_2
            Satel_analytics = pd.concat([Satel_analytics, df_1, df_2])
            Satel_analytics = Satel_analytics.drop_duplicates()
        return(Satel_analytics)

    def BaseAnalytics(self):
        print(4)
        """creates file named analytics, where speed, max acceleration (as defined by the local augmentation of a
        primary derivative of movement), extrapolates frequency and adds the unique identifier for the dataset,  """
        print("Running basic analysis...")
        plt.ioff()
        directory_periodogram = self.path_to_analysis_folder + "Periodogram/"
        directory_basic_analysis = self.path_to_analysis_folder + "BasicAnalysis/"
        try:
            os.makedirs(directory_periodogram)
            os.makedirs(directory_basic_analysis)
        except FileExistsError:
            pass
        analysis_table = pd.DataFrame()
        for file_table_name in self.satel_link["Satel Key ID"]:
            file_table = self.gait_data['locometre'][file_table_name]
            file_table.index = file_table['Temps ms']
            file_table = file_table[['pied gauche', 'pied droit']].apply(pd.to_numeric)
            file_table[['pied gauche non filtré', "pied droit non filtré"]] = file_table[['pied gauche', 'pied droit']]
            file_table["pied gauche"] = file_table["pied gauche"][file_table["pied gauche"]> file_table["pied gauche"].iloc[-1]*0.05]
            file_table["pied droit"] = file_table["pied droit"][file_table["pied droit"] > file_table["pied droit"].iloc[-1] * 0.05]
            file_table["pied gauche"] = file_table["pied gauche"][file_table["pied gauche"] < file_table["pied gauche"].iloc[-1]*0.90]
            file_table["pied droit"] = file_table["pied droit"][file_table["pied droit"] < file_table["pied droit"].iloc[-1] * 0.90]
            file_table["différence pied droit"] = file_table['pied droit'].diff()
            file_table["différence pied gauche"] = file_table['pied gauche'].diff()
            file_table = file_table / 1000
            file_table = file_table.reset_index(drop=True)
            arg_max_pied_gauche, _ = find_peaks(file_table["différence pied gauche"], distance=75)
            arg_max_pied_droit, _ = find_peaks(file_table["différence pied droit"], distance=75)
            arg_min_pied_gauche, _ = find_peaks(file_table["différence pied gauche"] * -1, distance=75)
            arg_min_pied_droit, _ = find_peaks(file_table["différence pied droit"] * -1, distance=75)
            prominence_max_pied_gauche = peak_prominences(file_table["différence pied gauche"], arg_max_pied_gauche)
            width_max_pied_gauche = peak_widths(file_table["différence pied gauche"], arg_max_pied_gauche,
                                                prominence_data=prominence_max_pied_gauche, rel_height=0.6)
            prominence_max_pied_droit = peak_prominences(file_table["différence pied droit"], arg_max_pied_droit)
            width_max_pied_droit = peak_widths(file_table["différence pied droit"], arg_max_pied_droit,
                                               prominence_data=prominence_max_pied_droit, rel_height=0.6)
            prominence_min_pied_gauche = peak_prominences(file_table["différence pied gauche"] * -1,
                                                          arg_min_pied_gauche)
            width_min_pied_gauche = peak_widths(file_table["différence pied gauche"] * -1, arg_min_pied_gauche,
                                                prominence_data=prominence_min_pied_gauche, rel_height=0.6)
            prominence_min_pied_droit = peak_prominences(file_table["différence pied droit"] * -1, arg_min_pied_droit)
            width_min_pied_droit = peak_widths(file_table["différence pied droit"] * -1, arg_min_pied_droit,
                                               prominence_data=prominence_min_pied_droit, rel_height=0.6)

            file_table["marche filtrée à l'initiation pied droit"] = file_table['pied droit'].iloc[
                                                                     arg_max_pied_droit[0]:arg_max_pied_droit[-1]]
            file_table["marche filtrée à l'initiation pied gauche"] = file_table['pied gauche'].iloc[
                                                                      arg_max_pied_gauche[0]:arg_max_pied_gauche[-1]]
            timedeltagauche = (arg_max_pied_gauche[-1] - arg_max_pied_gauche[0])/100
            timedeltadroit = (arg_max_pied_droit[-1] - arg_max_pied_droit[0])/100
            vitesse_pied_droit = (
                        file_table['pied droit'].iloc[arg_max_pied_droit[-1]] - file_table['pied droit'].iloc[
                    arg_max_pied_droit[0]]) / timedeltadroit
            vitesse_pied_gauche =  (
                        file_table['pied gauche'].iloc[arg_max_pied_gauche[-1]] - file_table['pied gauche'].iloc[
                    arg_max_pied_gauche[0]]) / timedeltagauche
            fréquence_pied_gauche = len(arg_max_pied_gauche) / timedeltagauche
            fréquence_pied_droit = len(arg_max_pied_droit) / timedeltadroit
            pct_phase_oscillante_gauche = (width_max_pied_droit[0].sum() + width_max_pied_gauche[0].sum()) / timedeltagauche
            mean_peak_droit = file_table['pied droit'][arg_max_pied_droit].mean()
            mean_peak_gauche = file_table['pied gauche'][arg_max_pied_gauche].mean()
            max_peak_droit = file_table['pied droit'][arg_max_pied_droit].max()
            max_peak_gauche = file_table['pied gauche'][arg_max_pied_gauche].max()
            mean_half_peak_width_droit = width_max_pied_droit[0].mean()
            mean_half_peak_width_gauche = width_max_pied_gauche[0].mean()
            analysis_table_ = pd.DataFrame(columns=["Vitesse moyenne pied droit", "Vitesse moyenne pied gauche",
                            "Accélération maximale moyenne pied droit", "Accélération maximale moyenne pied gauche",
                            "Accélération maximale pied droit", "Accélération maximale pied gauche",
                            "Fréquence moyenne pied droit", "Fréquence moyenne pied gauche",
                            "Temps cumulé à mi pic d'acceleration droit",
                            "Temps cumulé à mi pic d'acceleration gauche"])
            analysis_table_.loc[file_table_name] = [vitesse_pied_droit, vitesse_pied_gauche,
                                                                              mean_peak_droit, mean_peak_gauche,
                                                                              max_peak_droit, max_peak_gauche,
                                                                              fréquence_pied_droit,
                                                                              fréquence_pied_gauche,
                                                                              mean_half_peak_width_droit,
                                                                              mean_half_peak_width_gauche]
            analysis_table_ = analysis_table_.reset_index()
            analysis_table_ = pd.concat( [self.satel_link[self.satel_link["Satel Key ID"] == file_table_name].reset_index(),
                                         analysis_table_], axis=1, ignore_index=True)
            analysis_table = pd.concat([analysis_table, analysis_table_])
            # TODO if someone look at this, please change to ASCII charcter for publication... just trying to cut corners...
            if self.path_to_analysis_folder is not None:
                file_table_name = file_table_name.replace("/", "-")
                # sig = file_table["différence pied droit"]
                # p = Spectrogram(sig, ws=1, W=500, sampling=100)
                # p.periodogram()
                # p.plot()
                # plt.savefig(directory_periodogram + "figurePeriodogram_droit" + file_table_name + ".png", dpi = 300)
                # sig = file_table["différence pied gauche"]
                # p = Spectrogram(sig, ws=1, W=500, sampling=100)
                # p.periodogram()
                # p.plot()
                # plt.savefig(directory_periodogram + "figurePeriodogram_gauche" + file_table_name + ".png", dpi = 300)
                # plt.close('all')
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
                fig.subplots_adjust(hspace=0.5)
                line1, = ax1.plot(file_table.index, file_table['pied droit'])
                line2, = ax1.plot(file_table.index, file_table['pied gauche'])
                line3, = ax2.plot(file_table.index, file_table['différence pied droit'])
                line4, = ax2.plot(file_table.index, file_table['différence pied gauche'])
                line5, = ax2.plot(file_table['différence pied gauche'].iloc[arg_max_pied_gauche].index,
                                  file_table['différence pied gauche'].iloc[arg_max_pied_gauche], "+", color='tab:brown')
                line6, = ax2.plot(file_table['différence pied droit'].iloc[arg_max_pied_droit].index,
                                  file_table['différence pied droit'].iloc[arg_max_pied_droit], "+", color='tab:red')
                line7, = ax2.plot(file_table['différence pied gauche'].iloc[arg_min_pied_gauche].index,
                                  file_table['différence pied gauche'].iloc[arg_min_pied_gauche], "+", color='tab:blue')
                line8, = ax2.plot(file_table['différence pied droit'].iloc[arg_min_pied_droit].index,
                                  file_table['différence pied droit'].iloc[arg_min_pied_droit], "+", color='tab:cyan')
                line9, = ax3.plot(file_table["marche filtrée à l'initiation pied gauche"].index,
                                  file_table["marche filtrée à l'initiation pied gauche"])
                line10, = ax3.plot(file_table["marche filtrée à l'initiation pied droit"].index,
                                   file_table["marche filtrée à l'initiation pied droit"])
                for l, y, x1, x2 in zip(width_max_pied_droit[0], width_max_pied_droit[1], width_max_pied_droit[2],
                                        width_max_pied_droit[3]):
                    ax2.hlines(y=y,
                               xmin=file_table['différence pied droit'].index[x1.astype(int)],
                               xmax=file_table['différence pied droit'].index[x2.astype(int)], linewidth=2, color='r')
                    ax1.set_xlabel('time')
                ax1.set_ylabel('droite et gauche')
                ax1.grid(True)
                ax2.set_xlabel('time')
                ax2.set_ylabel('différences droite et gauche')
                ax2.grid(True)
                ax3.set_xlabel('time')
                ax3.set_ylabel('droite et gauche')
                ax3.grid(True)
                fig.savefig(directory_basic_analysis+"GaitDataAnalytics" + file_table_name + ".png", dpi=300)
                plt.close('all')
        analysis_table.columns = ["index", "Nom", "Avant/Après", "Satel Key ID",
                                  'Nouveau groupes étiologiques',
                                  'Nouvelles classes étiologiques',
                                  'Etiologies (grands groupes)', "Satel Key ID bis",
                                  "Vitesse moyenne pied droit", "Vitesse moyenne pied gauche",
                                  "Accélération maximale moyenne pied droit",
                                  "Accélération maximale moyenne pied gauche",
                                  "Accélération maximale pied droit", "Accélération maximale pied gauche",
                                  "Fréquence moyenne pied droit", "Fréquence moyenne pied gauche",
                                  "Temps cumulé à mi pic d'acceleration droit",
                                  "Temps cumulé à mi pic d'acceleration gauche"]
        return(analysis_table[["Nom", "Avant/Après", "Satel Key ID",
                                         'Nouveau groupes étiologiques',
                                         'Nouvelles classes étiologiques',
                                         'Etiologies (grands groupes)',
                          "Vitesse moyenne pied droit", "Vitesse moyenne pied gauche",
                            "Accélération maximale moyenne pied droit", "Accélération maximale moyenne pied gauche",
                            "Accélération maximale pied droit", "Accélération maximale pied gauche",
                            "Fréquence moyenne pied droit", "Fréquence moyenne pied gauche",
                            "Temps cumulé à mi pic d'acceleration droit",
                            "Temps cumulé à mi pic d'acceleration gauche"]])

    def CardiovascularRisk(self):
        PMH = pd.read_csv(self.path_to_csv_PMH)
        FDRCVindexAll = list()
        PMHdf = pd.DataFrame(columns=PMH.loc[:, "PMHx"].unique(), index=PMH.loc[:, "NAME SURNAME"].unique())
        for name in self.patient_info[
                               'matrice des données exploitables avant - après PL'].loc[:, "Name"].unique():
            table_pmh = PMH.loc[PMH["NAME SURNAME"] == name, :]
            FDRCVindex = table_pmh.loc[:, "PMHx"].str.count(
                "HCT|HTA|DT2|Coronaropathie|Neurovasculaire|Obésité|Exogénose|Tabac|aort|AOMI|nsuffisance renale").sum()
            FDRCVindexAll.append(int(FDRCVindex))
            for item in table_pmh.loc[:, "PMHx"].unique():
                if item in PMH.loc[:,"PMHx"].unique():
                    PMHdf.loc[name, item] = 1
                else:
                    PMHdf.loc[name, item] = 0
        self.patient_info['matrice des données exploitables avant - après PL']["FdRCV"] = FDRCVindexAll
        return PMHdf

    def MedsDuringTest(self):
        meds = pd.read_csv(self.path_to_MedsCSV)
        medsdf = pd.DataFrame(columns=meds.loc[:, "Groupe"].unique(), index=meds.loc[:, "NAME SURNAME"].unique())
        for name in self.patient_info[
                               'matrice des données exploitables avant - après PL'].loc[:, "Name"].unique():
            table_meds = meds.loc[meds["NAME SURNAME"] == name, :]
            for item in table_meds.loc[:,"Groupe"].unique():
                if item in meds.loc[:, "Groupe"].unique():
                    medsdf.loc[name, item] = 1
                else:
                    medsdf.loc[name, item] = 0
        return medsdf

    def MMSTest(self):
        mms = pd.read_csv(self.path_to_mmsCSV)
        mms["DATE"] = pd.to_datetime(mms["DATE"], dayfirst=True, errors="coerce")
        mms["NAME SURNAME"] = mms["NAME SURNAME"].str.replace("é", "e").str.lstrip(" ")
        mmsDf = pd.DataFrame(columns = ["Name", "MMS before", "MMS after", "time before", "time after", "date before", "date after"])
        mmsDfunique = pd.DataFrame(columns=["Name", "MMS", "Date"])
        for name, pivot in zip(self.patient_info[
                               'matrice des données exploitables avant - après PL'].loc[:, "Name"].unique(), self.patient_info[
                               'matrice des données exploitables avant - après PL'].loc[:, "Last"].unique()):
            table_mms = mms.loc[mms["NAME SURNAME"] == name, :]
            if len(table_mms) == 1:
                mmsDfunique.loc[name, : ] = [name, table_mms["RESULT"].values[0], table_mms["DATE"].values[0]]
            if len(table_mms) >1:
                try:
                    deltabefore, before, deltaafter, after = nearest2(table_mms["DATE"], pivot= pivot)
                    if deltabefore.days <100 and deltaafter.days <100:
                        mmsDf.loc[name,:] = [name,
                                   table_mms.loc[table_mms["DATE"]== before,"RESULT"].values[0],
                                    table_mms.loc[table_mms["DATE"]== after,"RESULT"].values[0],
                                        deltabefore, deltaafter, before, after]
                except ValueError:
                    print(name)
        return {"MMS before and after": mmsDf,"Unique MMSEs" : mmsDfunique }

    def secondary_analysis_routine(self):
        print("Secondary analysis...")
        self.secondary_analysis_routine = {"Analyse comparée" :self.comparative_analysis(),
                                           "Analyse spectrale": self.FourierFinder()}
        self.secondary_analysis_multivariate = self.MultivariateAnalysis()
    def comparative_analysis(self):
        print("Comparison of before/after data")
        divide_comp = pd.DataFrame()
        diff_comp = pd.DataFrame()
        after_df = pd.DataFrame()
        before_df = pd.DataFrame()

        for name in self.analytics["Nom"].drop_duplicates():
            features_str = ['Nom', 'Avant/Après', 'Satel Key ID', 'Nouveau groupes étiologiques',
                            'Nouvelles classes étiologiques', 'Etiologies (grands groupes)']
            features_maths = ['Vitesse moyenne pied droit', 'Vitesse moyenne pied gauche',
                              'Accélération maximale moyenne pied droit', 'Accélération maximale moyenne pied gauche',
                              'Accélération maximale pied droit', 'Accélération maximale pied gauche',
                              'Fréquence moyenne pied droit', 'Fréquence moyenne pied gauche',
                              "Temps cumulé à mi pic d'acceleration droit",
                              "Temps cumulé à mi pic d'acceleration gauche"]
            timestamp1 = self.analytics[(self.analytics["Nom"] == name) &
                                        (self.analytics["Avant/Après"] == "Avant")][features_str]
            timestamp2 = self.analytics[(self.analytics["Nom"] == name) &
                                        (self.analytics["Avant/Après"] == "Après")][features_str]
            timestamp1["Satel Key ID"] = timestamp1["Satel Key ID"].str.slice(stop=-9)
            timestamp2["Satel Key ID"] = timestamp2["Satel Key ID"].str.slice(stop=-9)
            timestamp1 = timestamp1.drop_duplicates()
            timestamp2 = timestamp2.drop_duplicates()
            math_frame1 = self.analytics[(self.analytics["Nom"] == name) &
                                         (self.analytics["Avant/Après"] == "Avant")][features_maths]
            math_frame2 = self.analytics[(self.analytics["Nom"] == name) &
                                         (self.analytics["Avant/Après"] == "Après")][features_maths]
            if len(math_frame1.index) > 1:
                math_frame1 = math_frame1.mean()
            else:
                math_frame1 = math_frame1.reset_index(drop=True).iloc[0]
            if len(math_frame2.index) > 1:
                math_frame2 = math_frame2.mean()
            else:
                math_frame2 = math_frame2.reset_index(drop=True).iloc[0]
            df_before = pd.DataFrame(math_frame1).transpose().reset_index(drop=True)
            df_after = pd.DataFrame(math_frame2).transpose().reset_index(drop=True)
            framediff = math_frame2 - math_frame1
            framediff = pd.DataFrame(framediff).transpose()
            framediff = pd.concat([timestamp1.reset_index(drop=True), framediff], axis=1)
            framediv = pd.DataFrame(math_frame2.reset_index(drop=True) / math_frame1.reset_index(drop=True)).transpose()
            framediv = pd.concat([timestamp1.reset_index(drop=True), framediv], axis=1)
            divide_comp = divide_comp.append(framediv, ignore_index=True)
            diff_comp = diff_comp.append(framediff, ignore_index=True)
            df_before = pd.concat([timestamp1.reset_index(drop=True), df_before], axis=1)
            df_after = pd.concat([timestamp2.reset_index(drop=True), df_after], axis=1)
            before_df = before_df.append(df_before, ignore_index=True)
            after_df = after_df.append(df_after, ignore_index=True)
        divide_comp.columns = features_str + features_maths
        diff_comp.columns = features_str + features_maths
        return({"Comparaison: division (après:avant)": divide_comp,
                "Comparaison: soustraction (après - avant)": diff_comp,
                "DF avant": before_df, "DF_apres": after_df })

    def FourierFinder(self):
        """creates file named fourier_analytics, where top frequencies by foot in the 0-2 Hz band is reported
        and comparatives (division of left foot by right foot) are registered
        and adds the prior dataframe for all  to it """
        print("Running spectral analysis: TTF and (if enabled) spectrogram analysis...")
        plt.ioff()
        directory_periodogram = self.path_to_analysis_folder + "/Periodogram/"
        directory_basic_analysis = self.path_to_analysis_folder + "/BasicAnalysis/"
        try:
            os.makedirs(directory_periodogram)
            os.makedirs(directory_basic_analysis)
        except FileExistsError:
            pass
        analysis_table = pd.DataFrame()
        for file_table_name in self.satel_link["Satel Key ID"]:
            file_table = self.gait_data['locometre'][file_table_name]
            file_table.index = file_table['Temps ms']
            file_table = file_table[['pied gauche', 'pied droit']].apply(pd.to_numeric)
            file_table[['pied gauche non filtré', "pied droit non filtré"]] = file_table[['pied gauche', 'pied droit']]
            file_table[["pied gauche", "pied droit"]] = file_table[["pied gauche", "pied droit"]][
                file_table["pied gauche"] > file_table["pied gauche"].iloc[-1] * 0.05]
            file_table[["pied gauche", "pied droit"]] = file_table[["pied gauche", "pied droit"]][
                file_table["pied gauche"] < file_table["pied gauche"].iloc[-1] * 0.90]
            file_table["différence pied droit"] = file_table['pied droit'].diff()
            file_table["différence pied gauche"] = file_table['pied gauche'].diff()
            sig = file_table["différence pied droit"]
            N = (sig.dropna().index.size)
            sig_normalized = sig.dropna() / sig.max()
            yf = fft(sig_normalized.to_numpy())
            xf = fftfreq(N, 1 / 100)
            sig = file_table["différence pied gauche"]
            sig_normalized = sig.dropna() / sig.max()
            yf2 = fft(sig_normalized.to_numpy())
            xf2 = fftfreq(N, 1 / 100)

            yf_diff = np.abs(yf2) - np.abs(yf)
            yf_div = np.abs(yf2) / np.abs(yf)
            arg_max_fourier_gauche, _ = find_peaks(np.abs(yf2), distance=20)
            arg_max_fourier_droit, _ = find_peaks(np.abs(yf), distance=20)
            arg_max_fourier_div, _ = find_peaks(np.abs(yf_div), distance=20)

            prominence_max_fourier_gauche = peak_prominences(np.abs(yf2), arg_max_fourier_gauche)
            width_max_pied_gauche = peak_widths(np.abs(yf2), arg_max_fourier_gauche,
                                                prominence_data=prominence_max_fourier_gauche, rel_height=0.9)
            prominence_max_fourier_droit = peak_prominences(np.abs(yf), arg_max_fourier_droit)
            width_max_pied_droit = peak_widths(np.abs(yf), arg_max_fourier_droit,
                                               prominence_data=prominence_max_fourier_droit, rel_height=0.9)
            peak_width_gauche_2Hz = [i for i in xf2[width_max_pied_gauche[3].astype(int)] if i < 2 and i > 0][0] - \
                                    [i for i in xf2[width_max_pied_gauche[2].astype(int)] if i < 2 and i > 0][0]
            peak_height_gauche_2Hz = max(np.abs(yf2[arg_max_fourier_gauche]))
            peak_width_droit_2Hz = [i for i in xf[width_max_pied_droit[3].astype(int)] if i < 2 and i > 0][0] - \
                                   [i for i in xf[width_max_pied_droit[2].astype(int)] if i < 2 and i > 0][0]
            peak_height_droit_2Hz = max(np.abs(yf[arg_max_fourier_droit]))
            peak_height_div = max(np.abs(yf_div[arg_max_fourier_div]))

            analysis_table_ = pd.DataFrame(columns=["Acceleration maximale droit en transformée de fourrier dans la bande 0-2Hz",
                                                    "Acceleration maximale gauche en transformée de fourrier dans la bande 0-2Hz",
                                                    "Variation du pas droit", "Variation du pas gauche",
                                                    "maximale de différentielle gauche/droite"])
            analysis_table_.loc[file_table_name] = [peak_width_droit_2Hz, peak_width_gauche_2Hz, peak_height_droit_2Hz,
                                                    peak_height_gauche_2Hz, peak_height_div]
            analysis_table_ = analysis_table_.reset_index()
            analysis_table_ = pd.concat( [self.satel_link[self.satel_link["Satel Key ID"] == file_table_name].reset_index(),
                                         analysis_table_], axis=1, ignore_index=True)
            analysis_table = pd.concat([analysis_table, analysis_table_])

            if self.path_to_analysis_folder is not None:
                file_table_name = file_table_name.replace("/", "-")
                fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
                fig.subplots_adjust(hspace=1)
                ax1.plot(xf, np.abs(yf))
                ax2.plot(xf2, np.abs(yf2))
                for l, y, x1, x2 in zip(width_max_pied_gauche[0], width_max_pied_gauche[1], width_max_pied_gauche[2],
                                        width_max_pied_gauche[3]):
                    ax2.hlines(y=y,
                               xmin=xf2[x1.astype(int)],
                               xmax=xf2[x2.astype(int)], linewidth=2, color='r')
                ax3.plot(xf2, np.abs(yf_diff))
                ax4.plot(xf2, np.abs(yf_div))
                ax1.plot(xf[arg_max_fourier_droit], np.abs(yf[arg_max_fourier_droit]), "+", color='tab:brown')
                ax2.plot(xf2[arg_max_fourier_gauche], np.abs(yf2[arg_max_fourier_gauche]), "+", color='tab:brown')
                ax4.plot(xf2[arg_max_fourier_div], yf_div[arg_max_fourier_div], "+", color='tab:brown')
                ax1.set_xlim(0, 2)
                ax2.set_xlim(0, 2)
                ax1.set_ylabel('FFT ')
                ax1.grid(True)
                ax1.set_title('FFT droite', fontsize = 5)
                ax2.set_title('FFT gauche', fontsize= 5)
                ax2.set_xlabel('Frequence (Hz)')
                ax2.set_ylabel('FFT')
                ax2.grid(True)
                ax3.set_xlim(0, 2)
                ax3.set_ylabel('FFT ')
                ax3.grid(True)
                ax3.set_title('FFT difference', fontsize= 5)
                ax4.set_title('FFT gauche/ FFT droite', fontsize= 5)
                ax4.set_xlim(0, 2)
                ax4.set_ylabel('FFT ')
                ax4.grid(True)
                fig.savefig(directory_periodogram+"GaitDataAnalytics" + file_table_name + ".png", dpi=300)
                plt.close('all')
        analysis_table.columns = ["index", "Nom", "Avant/Après", "Satel Key ID",
                                  'Nouveau groupes étiologiques',
                                  'Nouvelles classes étiologiques',
                                  'Etiologies (grands groupes)', "Satel Key ID bis",
                                  "Acceleration maximale droit en transformée de fourrier dans la bande 0-2Hz",
                                  "Acceleration maximale gauche en transformée de fourrier dans la bande 0-2Hz",
                                  "Variation du pas droit", "Variation du pas gauche",
                                  "maximale de différentielle gauche/droite"]
        features_ttf = [ "Acceleration maximale droit en transformée de fourrier dans la bande 0-2Hz",
                               "Acceleration maximale gauche en transformée de fourrier dans la bande 0-2Hz",
                               "Variation du pas droit", "Variation du pas gauche",
                               "maximale de différentielle gauche/droite"]
        features_str = ['Nom', 'Avant/Après', 'Satel Key ID', 'Nouveau groupes étiologiques',
                        'Nouvelles classes étiologiques', 'Etiologies (grands groupes)']
        features_maths = ['Vitesse moyenne pied droit', 'Vitesse moyenne pied gauche',
                          'Accélération maximale moyenne pied droit', 'Accélération maximale moyenne pied gauche',
                          'Accélération maximale pied droit', 'Accélération maximale pied gauche',
                          'Fréquence moyenne pied droit', 'Fréquence moyenne pied gauche',
                          "Temps cumulé à mi pic d'acceleration droit",
                          "Temps cumulé à mi pic d'acceleration gauche"]

        analysis_table = analysis_table[["Nom", "Avant/Après", "Satel Key ID",
                               'Nouveau groupes étiologiques',
                               'Nouvelles classes étiologiques',
                               'Etiologies (grands groupes)'] + features_ttf]

        print("-comparison of TTF results")
        divide_comp = pd.DataFrame()
        diff_comp = pd.DataFrame()
        for name in analysis_table["Nom"].drop_duplicates():
            timestamp1 = analysis_table[(analysis_table.loc[:, "Nom"] == name) &
                                        (analysis_table.loc[:, "Avant/Après"] == "Avant")].loc[:, features_str]
            timestamp1["Satel Key ID"] = timestamp1["Satel Key ID"].str.slice(stop=-9)
            timestamp1 = timestamp1.drop_duplicates()
            math_frame1 = analysis_table[(analysis_table.loc[:, "Nom"] == name) &
                                        (analysis_table.loc[:, "Avant/Après"] == "Avant")].loc[:, features_ttf]
            math_frame2 = analysis_table[(analysis_table.loc[:, "Nom"] == name) &
                                        (analysis_table.loc[:, "Avant/Après"] == "Après")].loc[:, features_ttf]
            if len(math_frame1.index) > 1:
                math_frame1 = math_frame1.mean()
            else:
                math_frame1 = math_frame1.reset_index(drop=True).iloc[0]
            if len(math_frame2.index) > 1:
                math_frame2 = math_frame2.mean()
            else:
                math_frame2 = math_frame2.reset_index(drop=True).iloc[0]
            framediff = math_frame2 - math_frame1
            framediff = pd.DataFrame(framediff).transpose()
            framediff = pd.concat([timestamp1.reset_index(drop=True), framediff], axis=1)
            framediv = pd.DataFrame(math_frame2.reset_index(drop=True) / math_frame1.reset_index(drop=True)).transpose()
            framediv = pd.concat([timestamp1.reset_index(drop=True), framediv], axis=1)
            divide_comp = divide_comp.append(framediv, ignore_index=True)
            diff_comp = diff_comp.append(framediff, ignore_index=True)
        divide_comp.columns = features_str + features_ttf
        diff_comp.columns = features_str + features_ttf
        return({"Comparaison TTF: division (après:avant)": divide_comp,
                "Comparaison TTF: soustraction (après - avant)": diff_comp,
                "Analyse tout échantillons TTF": analysis_table})

    def MultivariateAnalysis(self):
        print("Multivariate analysis")
        directory_multivariate_analysis = self.path_to_analysis_folder + "MultivariateAnalysis/"
        try:
            os.makedirs(directory_multivariate_analysis)
        except FileExistsError:
            pass
        features_str = ['Nom', 'Avant/Après', 'Satel Key ID', 'Nouveau groupes étiologiques',
                        'Nouvelles classes étiologiques', 'Etiologies (grands groupes)']
        features_linear = ['Vitesse moyenne pied droit', 'Vitesse moyenne pied gauche',
                          'Accélération maximale moyenne pied droit', 'Accélération maximale moyenne pied gauche',
                          'Accélération maximale pied droit', 'Accélération maximale pied gauche',
                          'Fréquence moyenne pied droit', 'Fréquence moyenne pied gauche',
                          "Temps cumulé à mi pic d'acceleration droit", "Temps cumulé à mi pic d'acceleration gauche"]
        features_fournier = ["Acceleration maximale droit en transformée de fourrier dans la bande 0-2Hz",
                               "Acceleration maximale gauche en transformée de fourrier dans la bande 0-2Hz",
                               "Variation du pas droit", "Variation du pas gauche",
                               "maximale de différentielle gauche/droite"]
        dfstr = self.secondary_analysis_routine["Analyse comparée"][
            "Comparaison: division (après:avant)"].loc[:, features_str]
        dfdiv = self.secondary_analysis_routine["Analyse comparée"][
                        "Comparaison: division (après:avant)"].loc[
                    :, features_linear].add_suffix(' comparaison par division')
        dfdiff = self.secondary_analysis_routine["Analyse comparée"]["Comparaison: soustraction (après - avant)"].loc[
                         :, features_linear].add_suffix(' comparaison des différences')
        dfFournierdiv = self.secondary_analysis_routine[
            "Analyse spectrale"]["Comparaison TTF: division (après:avant)"].loc[
                         :, features_fournier].add_suffix(' (comparaison division en TTF)')
        dfFournierdiff = self.secondary_analysis_routine[
            "Analyse spectrale"][
            "Comparaison TTF: soustraction (après - avant)"].loc[
                         :, features_fournier].add_suffix(' (comparaison différence en TTF)')
        dfall = pd.concat([dfstr, dfdiff, dfdiv, dfFournierdiv, dfFournierdiff], axis=1)
        dfall = dfall.loc[:, dfall.columns.drop_duplicates()]
        y = dfall.loc[:, 'Nouveau groupes étiologiques'].values
        x = dfall.loc[:, ~dfall.columns.isin(features_str)].values
        x[~np.isfinite(x)] = 0
        x = StandardScaler().fit_transform(x)
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(x)
        principalDf = pd.DataFrame(data=principalComponents
                                   , columns=['principal component 1', 'principal component 2'])
        finalDf = pd.concat([principalDf, dfall[['Nouveau groupes étiologiques']]], axis=1, ignore_index=True)
        finalDf.columns = ['principal component 1', 'principal component 2', 'Nouveau groupes étiologiques']

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Principal Component 1', fontsize=15)
        ax.set_ylabel('Principal Component 2', fontsize=15)
        ax.set_title('2 component PCA', fontsize=20)
        targets = set(dfall['Nouveau groupes étiologiques'])
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
                  'tab:olive', 'tab:cyan', 'black']
        for target, color in zip(targets, colors):
            indicesToKeep = finalDf['Nouveau groupes étiologiques'] == target
            ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                       , finalDf.loc[indicesToKeep, 'principal component 2']
                       , c=color
                       , s=50, alpha=0.5)
        ax.legend(targets)
        ax.grid()
        fig.savefig(directory_multivariate_analysis +"GaitDataAnalyticsPCAafterFT" + ".png", dpi=300)
        return finalDf









