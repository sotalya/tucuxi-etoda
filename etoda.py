import os
import re
from bs4 import BeautifulSoup
import sotalya.pycli as module
import sotalya.data.requests as R
from sotalya.importexport.exporttqf import ExportTqf
import sotalya.data.query as Q
from sotalya.tucuxi.utils import str_to_datetime, str_to_time
from utils import display_computing_query_response
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
import pickle
import pandas as pd
import copy
from typing import List
from argparse import ArgumentParser
import argparse
import ast
from matplotlib.pyplot import MultipleLocator
import matplotlib.ticker as ticker
import seaborn as sns
from scipy.interpolate import interp1d
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")


fontsize_fig = 40
plt.rc('font', family = 'Times New Roman')



parser = ArgumentParser(description = '3D_errorgrid',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
basepath = os.path.dirname(__file__)




def parse_str_list(str_list):
    return [item.strip() for item in str_list.replace(' ', ',').split(',') if item.strip()]


def parse_list(list_str):
    try:
        result = ast.literal_eval(list_str)
        
        if isinstance(result, list):
            return result
        else:
            raise ValueError
    except:
        raise argparse.ArgumentTypeError("Invalid format for list. Expected format: \" " + str([1, 2, 3]) + '\"')


def parse_nested_list(nested_list_str, num):
    try:
        result = ast.literal_eval(nested_list_str)
        if isinstance(result, list) and all(isinstance(i, list) and len(i) == num for i in result):
            return result
        else:
            raise ValueError
    except:
        raise argparse.ArgumentTypeError("Invalid format for nested list. Expected format: [[0.05, -700], [0.05, -1200]]")



def parse_the_args():
    drug_folder_path = basepath + '/data_input/drugfiles/'
    patient_example_path = basepath + '/data_input/ch.tucuxi.imatinib.gotta2012.2.tqf'
    query_example_path = basepath + '/templates/query_template.tqf'


    # (1) imatinib (target: residual)
    drugmodel = 'ch.tucuxi.imatinib.gotta2012.tdd'
    # (2) vancomycin (target: residual)
    # drugmodel = 'ch.tucuxi.vancomycin.liu2019.tdd'
    # (3) vancomycin (target: auc/mic)
    # drugmodel = 'ch.tucuxi.vancomycin.aucliu2019.tdd'

    
    
    date_input = '2024-02-16T08:00:00'
    dosage_date = '2024-03-01T08:00:00'

    dosage_duration = '24'
    dosage_adjustment_duration = '24'

    sample_date = '2024-03-19T08:00:00'
    percentiles = [1, 99]
   


    # 有关结果图的时间点数的参数
    parser.add_argument('--num_con', type = int, default = 10,
                        help = 'How many concentrations in each axis should be considered in the 3D_errorgrid')
    parser.add_argument('--nb_points_per_hour', type = int, default = 20,
                        help = 'How many points in the concentration-time evolution per hour')
    parser.add_argument('--hours', type = parse_list, default = [0, 2], 
                        help = 'The hours to draw a new 3D_errorgrid after a new dose in the steady state. Expected format: \"' + str([1, 2, 3]) + '\"')   
    parser.add_argument('--percentiles', type = parse_list, default = percentiles,
                        help = 'The range of percentiles to calculate distribution and plot range')       
    # 有关想要实现的功能
    parser.add_argument('--measurement_error', type = bool, default = 1, help = 'The construction of error grid to show measurement error tolerance.')
    parser.add_argument('--measurement_error_distribution_type', type = str, default = 'uniform')
    # 有关绘制全局图或者局部图
    parser.add_argument('--plot_full_matrix', type = bool, default = 1, help = 'The flag to plot full matrix or not.')
    parser.add_argument('--mea_range', type = parse_list, default = [32000, 35000], help = 'The measured concentration range of sub-plot.')
    parser.add_argument('--true_range', type = parse_list, default = [5000, 15000], help = 'The true concentration range of sub-plot.')
    parser.add_argument('--unit_change_flag', type = bool, default = 0, help = 'The flag for unit changing.')
    parser.add_argument('--unit_change_to', type = str, default = 'mg/l')

    # 有关初始dosage的参数
    parser.add_argument('--drug_folder_path', type = str, default = drug_folder_path,
                        help = 'The path of the drug files folder.')
    parser.add_argument('--drug_model_path', type = str, default = drugmodel,
                        help = 'The name of the drug model.')
    parser.add_argument('--patient_example_path', type = str, default = patient_example_path,
                        help = 'The path of the patient example.')
    parser.add_argument('--query_example_path', type = str, default = query_example_path,
                        help = 'The path of the query example.')
    parser.add_argument('--date_input', type = str, default = date_input,
                        help = 'The date of results generation.')
    # Expected format: YYYY-MM-DDTHH:MM:SS
    parser.add_argument('--dosage_date', type = str, default = dosage_date,
                        help = 'The date to start the dosage.')
    parser.add_argument('--dosage_duration', type = str, default = dosage_duration,
                        help = 'The duration of the existing dosage.')
    parser.add_argument('--dosage_adjustment_duration', type = str, default = dosage_adjustment_duration,
                        help = 'The duration of the new dosage.')
    parser.add_argument('--sample_date', type = str, default = sample_date,
                        help = 'The date for the first sample.')
    parser.add_argument('--target_value_select', type = str, default = 'automatic',
                        help = 'The value used for target setting. (automatic: use information in drug file; manual: use information by input)')
    
    # 三维图绘制参数
    parser.add_argument('--plot_3d', type = bool, default = 1, help = 'Plot 3D figure')

    return parser

 


def check_attribute(example_object):
    all_attributes = dir(example_object)

    # 打印对象的所有属性和对应的值
    for attribute in all_attributes:
        # 使用 getattr() 获取属性值
        attribute_value = getattr(example_object, attribute)
        print(f"{attribute}: {attribute_value}")


def convert_time(value: str, unit: str) -> str:
    if unit == 'h':
        hours = int(value)
        minutes = 0
        seconds = 0
    elif unit == 'min':
        hours = 0
        minutes = int(value)
        seconds = 0
    return f"{hours:02}:{minutes:02}:{seconds:02}"


class Default_dose:
    value: float
    unit: str
    infusionTimeInMinutes: timedelta

    def __init__(self, formulationAndRoute):
        self.value = float(formulationAndRoute.dosages.availableDoses.find('default').standardValue.string)
        self.unit = formulationAndRoute.dosages.availableDoses.unit.string
        # oral的情况没有这个参数
        if formulationAndRoute.administrationRoute.string == 'oral':
            self.infusionTimeInMinutes = str_to_time('1:00:00')
        else:
            default_infusion_unit = formulationAndRoute.dosages.availableInfusions.unit.string
            default_infusion_value = float(formulationAndRoute.dosages.availableInfusions.find('default').standardValue.string)
            self.infusionTimeInMinutes = str_to_time(convert_time(default_infusion_value, default_infusion_unit))


class Default_formulationAndRoutes:
    formulation: str
    administrationName: str
    administrationRoute: str
    absorptionModel: str

    def __init__(self, formulationAndRoute):
        self.formulation = formulationAndRoute.formulation.string
        self.administrationName = formulationAndRoute.administrationName.string
        self.administrationRoute = formulationAndRoute.administrationRoute.string
        self.absorptionModel = formulationAndRoute.absorptionModel.string


class Default_covariate:
    covariateId: str
    date: str
    value: str
    unit: str
    dataType: str
    nature: str

    def __init__(self, c, date_input):
        self.covariateId = c.covariateId.string
        self.date = str_to_datetime(date_input)
        self.value = c.covariateValue.standardValue.string
        self.unit = c.unit.string
        self.dataType = c.dataType.string
        if self.dataType == 'date':
            self.nature = 'discrete'
        elif self.dataType == 'double':
            self.nature = 'continuous'
        elif self.dataType == 'bool':
            self.nature = 'categorical'
        else:
            self.nature = '-'


class Default_covariates:
    coveriates: List[Default_covariate]
    def __init__(self, soup = None, date_input = None):
        self.coveriates = []
        if soup is not None:
            if soup.drugModel.covariates:
                for c in soup.drugModel.covariates.find_all('covariate'):
                    self.coveriates.append(Default_covariate(c, date_input))


class Drug_target:
    activeMoietyId: str
    targetType: str
    unit: str
    inefficacyAlarm: float
    min: float
    best: float
    max: float
    toxicityAlarm: float
    mic_unit: str
    mic: float


    def __init__(self, t, activeMoietyId):
        self.activeMoietyId = activeMoietyId
        self.targetType = t.targetType.string
        self.unit = t.targetValues.unit.string
        self.toxicityAlarm = float(t.targetValues.toxicityAlarm.standardValue.string)
        self.min = float(t.targetValues.min.standardValue.string)
        self.best = float(t.targetValues.best.standardValue.string)
        self.max = float(t.targetValues.max.standardValue.string)
        self.inefficacyAlarm = float(t.targetValues.inefficacyAlarm.standardValue.string)
        if self.targetType == 'auc24DividedByMic' or self.targetType == 'residualDividedByMic':
            self.mic_unit = t.targetValues.mic.unit.string
            self.mic = float(t.targetValues.mic.micValue.standardValue.string)
        else:
            self.mic = 0
            self.mic_unit= ''

             
class DrugInformation:
    drugId: str
    drugModelId: str
    activePrinciple: str
    brandName: str
    atc: str
    default_formulationAndRoutes: Default_formulationAndRoutes
    default_dose: Default_dose
    default_dose_interval: timedelta
    targets: List[Drug_target]


    def __init__(self, d = None):
        self.drugId = ''
        self.drugModelId = ''
        self.activePrinciple = ''
        self.brandName = ''
        self.atc = ''
        self.targets = []
        if d is not None:
            self.drugId = d.drugModel.drugId.string
            self.drugModelId = d.drugModel.drugModelId.string
            self.activePrinciple = 'something'
            self.brandName = 'somebrand'
            self.atc = 'something'

            if d.drugModel.find('formulationAndRoutes', {'default': 'id0'}) is not None:
                formulationAndRoute = d.drugModel.find('formulationAndRoutes', {'default': 'id0'}).formulationAndRoute
            else:
                if d.drugModel.find('formulationAndRoutes', {'default': 'id1'}) is not None:
                    formulationAndRoute = d.drugModel.find('formulationAndRoutes', {'default': 'id1'}).formulationAndRoute
                else:
                    print("FormulationAndRoutes is not found.")
            

            self.default_formulationAndRoutes = Default_formulationAndRoutes(formulationAndRoute)
            
    
            self.default_dose = Default_dose(formulationAndRoute)

            default_dose_interval_value = float(formulationAndRoute.dosages.availableIntervals.find('default').standardValue.string)
            default_dose_interval_unit = formulationAndRoute.dosages.availableIntervals.unit.string
            self.default_dose_interval = str_to_time(convert_time(default_dose_interval_value, default_dose_interval_unit))

            activeMoietyId = d.drugModel.activeMoieties.activeMoiety.activeMoietyId
            for t in d.drugModel.activeMoieties.activeMoiety.targets.find_all('target'):
                self.targets.append(Drug_target(t, activeMoietyId))



def fixed_target_sequence_produce(drug_input):
    colors = ['purple', 'blue', 'green', 'darkorange', 'red']
    target_min = drug_input.targets[0].min
    target_max = drug_input.targets[0].max
    target_inefficacyAlarm = drug_input.targets[0].inefficacyAlarm
    target_toxicityAlarm = drug_input.targets[0].toxicityAlarm
    threshold_range = [[0, target_inefficacyAlarm], [target_inefficacyAlarm, target_min], [target_min, target_max], [target_max, target_toxicityAlarm], [target_toxicityAlarm, float('inf')]]
    threshold_label = [-2, -1, 0, 1, 2]
    num_points = 10
    target_unit = drug_input.targets[0].unit
    target_type = drug_input.targets[0].targetType


    fig = plt.figure(figsize = (15, 10))
    ax1 = fig.add_subplot(111)
    for i in range(len(threshold_range)):
        if i == 0:
            input_x_list = np.linspace(0.5 * threshold_range[i][1], threshold_range[i][1], num_points).tolist()
        elif i == len(threshold_range) - 1:
            input_x_list = np.linspace(threshold_range[i][0], 1.5 * threshold_range[i][0], num_points).tolist()
        else:
            input_x_list = np.linspace(threshold_range[i][0], threshold_range[i][1], num_points).tolist()
        input_y_list = [i for _ in range(num_points)]
        
        p1, = plt.plot(input_x_list, input_y_list, color = colors[i])
    
    ax1.set_ylabel("Label", fontsize = fontsize_fig)    
    ax1.set_xlabel("%s"%(target_type) + ' (%s)'%(target_unit), fontsize = fontsize_fig)
    plt.title('%s'%(drug_input.drugId) , fontsize = fontsize_fig)

    # plt.legend(fontsize = fontsize_fig, loc = 0)
    y_major_locator = MultipleLocator(1)
    ax1.yaxis.set_major_locator(y_major_locator)

    for label in ax1.xaxis.get_ticklabels():
        label.set_fontsize(fontsize_fig)
    for label in ax1.yaxis.get_ticklabels():
        label.set_fontsize(fontsize_fig)
    plt.draw()
    plt.pause(2)

    return threshold_range, colors



def range_define(args, drug_input, query, dosage_date, dosage_duration):

    target_max = drug_input.targets[0].max
    target_min = drug_input.targets[0].min
    target_unit = drug_input.targets[0].unit
    target_type= drug_input.targets[0].targetType


    # threshold_range = [[0, target_inefficacyAlarm], [target_inefficacyAlarm, target_min], [target_min, target_max], [target_max, target_toxicityAlarm], [target_toxicityAlarm, float('inf')]]
    # threshold_label = [-2, -1, 0, 1, 2]

    # 先加入一个dosage
    query.drugs[0].dosageHistory.dosageTimeRanges = []
    dosage_history = []
    
    dosage_start_date = dosage_date
    dosage_end_date = dosage_date + dosage_duration
    lasting_dosage = Q.LastingDosage()
    
    lasting_dosage.interval = drug_input.default_dose_interval
    lasting_dosage.dose.infusionTimeInMinutes = drug_input.default_dose.infusionTimeInMinutes
    lasting_dosage.dose.unit = drug_input.default_dose.unit
    lasting_dosage.dose.value = drug_input.default_dose.value
    lasting_dosage.formulationAndRoute.absorptionModel = drug_input.default_formulationAndRoutes.absorptionModel
    lasting_dosage.formulationAndRoute.administrationRoute = drug_input.default_formulationAndRoutes.administrationRoute
    lasting_dosage.formulationAndRoute.administrationName = drug_input.default_formulationAndRoutes.administrationName
    lasting_dosage.formulationAndRoute.formulation = drug_input.default_formulationAndRoutes.formulation
    dosage_history_1 = Q.DosageTime.create_dosage_time_range(dosage_start_date, lasting_dosage, dosage_end_date)
    dosage_history.append(dosage_history_1)

   

    query.drugs[0].dosageHistory.dosageTimeRanges = dosage_history
    for i in range(len(query.drugs[0].dosageHistory.dosageTimeRanges)):
        query.drugs[0].dosageHistory.dosageTimeRanges[i].dosage.dose.infusionTimeInMinutes = query.drugs[0].dosageHistory.dosageTimeRanges[i].dosage.dose.infusionTimeInMinutes.total_seconds() / 60

    # 清除原有的sample
    query.drugs[0].samples = []

    # 增加一个看prior的request
    query.requests = []
    request = R.Request()
    request.requestId = 'apriori'
    request.drugId = drug_input.drugId
    request.drugModelId = drug_input.drugModelId
    computing_option = R.ComputingOption(R.ParametersTypeEnum.apriori, R.CompartmentOptionEnum.allActiveMoieties, True, True, True)
    # ranks = [5, 95]
    # ranks = [1, 99]
    ranks = args.percentiles
    request.computingTraits = R.PercentilesTraits.create_percentiles_traits(args.nb_points_per_hour, dosage_end_date - timedelta(days = 2), dosage_end_date, computing_option, ranks)
    query.requests.append(request)

    exporter = ExportTqf()
    prior_content = exporter.export_to_string(query, args.query_example_path)


    # print("Tqf Content is:")
    # print(prior_content)

    # print("=" * 50)



    prior_results = module.compute_tqf2object(prior_content, [args.drug_folder_path])
  

    # # 输出具体内容
    # print(f"Results are : {prior_results}")
    # display_computing_query_response(prior_results)
    

    assert prior_results.query_status == module.QueryStatus.ok
    for srd in prior_results.responses:
        assert srd.computing_response.computing_status == module.ComputingStatus.ok
    
    plot_maximum = - float('inf')
    plot_minimum = float('inf')



    prior_data = prior_results.responses[0].computing_response.data
    for (i, rank) in enumerate(prior_data.percentile_ranks):
        # print(f"======= Rank : {rank}")
        cycle_data_onerank = prior_data.cycle_data(i)
        cycle_data_list = []
        time_list = []
        index = 0
        for cycle in cycle_data_onerank:
            cycle_data_list += cycle.concentrations[0]

            time_list += [index + j for j in range(len(cycle.concentrations[0]))]
            index += len(cycle.concentrations[0])

        cycle_unit = cycle.unit.value

        max_now = max(cycle_data_list)
        min_now = min(cycle_data_list)
        if max_now > plot_maximum:
            plot_maximum = max_now
        if min_now < plot_minimum:
            plot_minimum = min_now


        
    return query, target_unit, cycle_unit, plot_maximum, plot_minimum


def distribution_error(args, con_list):
    
    if args.measurement_error_distribution_type == 'uniform':
        samples = np.random.uniform(-25, 25, 1000000)

        # samples = samples[(samples > -15) & (samples < 15)]

        fig = plt.figure(figsize = (15, 5))
        ax1 = fig.add_subplot(111)
        # KDE 分布图
        sns.kdeplot(samples,  bw_adjust = 3, clip=[-15, 15], shade = True)
        plt.title('Relative Error Distribution', fontsize = fontsize_fig)
        
        ax1.set_ylabel("Probability", fontsize = fontsize_fig) 
        
        ax1.set_xlabel("Relative Error (%)", fontsize = fontsize_fig)
        plt.yticks([])
        plt.xlim([-20, 20])
        for label in ax1.xaxis.get_ticklabels():
            label.set_fontsize(fontsize_fig)
        for label in ax1.yaxis.get_ticklabels():
            label.set_fontsize(fontsize_fig)
        plt.draw()
        plt.pause(2)
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(5))

        return 1
        
   
    elif args.measurement_error_distribution_type == 'all':
        samples = np.random.uniform(-15, 15, 10000000)

        fig = plt.figure(figsize = (15, 5))
        ax1 = fig.add_subplot(111)
        # KDE 分布图
        sns.kdeplot(samples,  bw_adjust = 1, shade = True)
        plt.title('Relative Error Distribution', fontsize = fontsize_fig)
        plot_line = sns.kdeplot(samples, bw_adjust = 1)
        fit_line = plot_line.lines[0]
        x_vals = fit_line.get_xdata()
        y_vals = fit_line.get_ydata()

        interp_func = interp1d(x_vals, y_vals, kind = 'cubic', fill_value= 'extrapolate')
        
        percentage = [-15, -10, -5, 0, 5, 10, 15]
        percentage_fit = [[p, interp_func(p)] for p in percentage]
        # print(np.array(percentage_fit)[:,1])
        plt.plot(percentage, np.array(percentage_fit)[:,1], marker = 'o', linestyle = '')
        
        ax1.set_ylabel("Probability", fontsize = fontsize_fig) 
        
        ax1.set_xlabel("Relative Error (%)", fontsize = fontsize_fig)
        plt.yticks([])
        plt.xlim([-20, 20])
        for label in ax1.xaxis.get_ticklabels():
            label.set_fontsize(fontsize_fig)
        for label in ax1.yaxis.get_ticklabels():
            label.set_fontsize(fontsize_fig)
        plt.draw()
        plt.pause(2)
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(5))

        return interp_func

    
        
    

def distribution_plot_priori(args, drug_input, query, dosage_date, dosage_duration, percentage_range):
    minimum_percentage = percentage_range[0]
    maximum_percentage = percentage_range[1]
    target_unit = drug_input.targets[0].unit

    query.drugs[0].dosageHistory.dosageTimeRanges = []
    dosage_history = []
    

    dosage_start_date = dosage_date
    dosage_end_date = dosage_date + dosage_duration
    lasting_dosage = Q.LastingDosage()
    # lasting_dosage.interval = str_to_time('24:00:00')
    # lasting_dosage.dose.infusionTimeInMinutes = str_to_time('1:00:00')
    # lasting_dosage.dose.unit = 'mg'
    # lasting_dosage.dose.value = float(400)
    # lasting_dosage.formulationAndRoute.absorptionModel = 'extravascular'
    # lasting_dosage.formulationAndRoute.administrationRoute = 'oral'
    # lasting_dosage.formulationAndRoute.administrationName = 'foo bar'
    # lasting_dosage.formulationAndRoute.formulation = 'parenteralSolution'
    lasting_dosage.interval = drug_input.default_dose_interval
    lasting_dosage.dose.infusionTimeInMinutes = drug_input.default_dose.infusionTimeInMinutes
    lasting_dosage.dose.unit = drug_input.default_dose.unit
    lasting_dosage.dose.value = drug_input.default_dose.value
    lasting_dosage.formulationAndRoute.absorptionModel = drug_input.default_formulationAndRoutes.absorptionModel
    lasting_dosage.formulationAndRoute.administrationRoute = drug_input.default_formulationAndRoutes.administrationRoute
    lasting_dosage.formulationAndRoute.administrationName = drug_input.default_formulationAndRoutes.administrationName
    lasting_dosage.formulationAndRoute.formulation = drug_input.default_formulationAndRoutes.formulation
    dosage_history_1 = Q.DosageTime.create_dosage_time_range(dosage_start_date, lasting_dosage, dosage_end_date)
    dosage_history.append(dosage_history_1)

    query.drugs[0].dosageHistory.dosageTimeRanges = dosage_history
    for i in range(len(query.drugs[0].dosageHistory.dosageTimeRanges)):
        query.drugs[0].dosageHistory.dosageTimeRanges[i].dosage.dose.infusionTimeInMinutes = query.drugs[0].dosageHistory.dosageTimeRanges[i].dosage.dose.infusionTimeInMinutes.total_seconds() / 60

    # 清除原有的sample
    query.drugs[0].samples = []

    # 增加一个看prior的request
    query.requests = []
    request = R.Request()
    request.requestId = 'distribution'
    request.drugId = drug_input.drugId
    request.drugModelId = drug_input.drugModelId
    computing_option = R.ComputingOption(R.ParametersTypeEnum.apriori, R.CompartmentOptionEnum.allActiveMoieties, True, True, True)
    ranks = [rank_select for rank_select in range(minimum_percentage, maximum_percentage + 1)]
    request.computingTraits = R.PercentilesTraits.create_percentiles_traits(args.nb_points_per_hour, dosage_end_date - drug_input.default_dose_interval, dosage_end_date, computing_option, ranks)
    query.requests.append(request)

    exporter = ExportTqf()
    prior_content = exporter.export_to_string(query, args.query_example_path)

    prior_results = module.compute_tqf2object(prior_content, [args.drug_folder_path])

    # 输出具体内容
    # print(f"Results are : {prior_results}")
    # display_computing_query_response(prior_results)

    assert prior_results.query_status == module.QueryStatus.ok
    for srd in prior_results.responses:
        assert srd.computing_response.computing_status == module.ComputingStatus.ok


    plot_maximum = -float('inf')
    plot_minimum = float('inf')



    hour_distribution_list = [[] for _ in range(len(args.hours))]
    hour_check = []
    interval = drug_input.default_dose_interval.total_seconds() / 3600
    for hour in args.hours:
        if hour < interval:
            hour_check.append(hour)
        else:
            while hour >= interval:
                hour -= interval
            hour_check.append(hour)


    cycle_all = []


    prior_data = prior_results.responses[0].computing_response.data
    for (i, rank) in enumerate(prior_data.percentile_ranks):
        # print(f"======= Rank : {rank}")
        cycle_data_onerank = prior_data.cycle_data(i)
        cycle_data_list = []
        time_list = []
        for cycle in cycle_data_onerank:
            cycle_data_list += cycle.concentrations[0]
            time_list += cycle.times[0]
        
        cycle_all.append(cycle_data_list)


        cycle_unit = cycle.unit.value

        max_now = max(cycle_data_list)
        min_now = min(cycle_data_list)
        if max_now > plot_maximum:
            plot_maximum = max_now
        if min_now < plot_minimum:
            plot_minimum = min_now

        for hour_index in range(len(args.hours)):
            idx = min(range(len(time_list)), key=lambda i: abs(time_list[i] - hour_check[hour_index]))
            # print("hhhh")
            # print(time_list[idx])
            # print(cycle_data_list[idx])
            hour_distribution_list[hour_index].append(cycle_data_list[idx])

    
    
    con_list = list(np.linspace(plot_minimum, plot_maximum, args.num_con))
    # print(con_list)
    con_select_prob_hour = []
    
    for hour_index in range(len(args.hours)):
        fig = plt.figure(figsize = (15, 5))
        ax1 = fig.add_subplot(111)
        # KDE 分布图
        sns.kdeplot(hour_distribution_list[hour_index], shade = True)

        plot_line = sns.kdeplot(hour_distribution_list[hour_index])
        fit_line = plot_line.lines[0]  
        x_vals = fit_line.get_xdata()
        y_vals = fit_line.get_ydata()

        interp_func = interp1d(x_vals, y_vals, kind='cubic', fill_value='extrapolate')
        
        con_select_prob = []
        for con in con_list:
            # if (con <= max(hour_distribution_list[hour_index]) and con >= min(hour_distribution_list[hour_index])):
            if interp_func(con) > 0:
                con_select_prob.append(interp_func(con))
            else:
                con_select_prob.append(0)

        con_select_prob_hour.append(con_select_prob)
                
        # con_select_prob = [[con, interp_func(con)] for con in con_list if (con <= max(hour_distribution_list[hour_index]) and con >= min(hour_distribution_list[hour_index]))]
        # plt.plot(con_list, con_select_prob, marker = 'o', linestyle = '')

        plt.title('Distribution (Sampling time: %d hours)' % (args.hours[hour_index]), fontsize = fontsize_fig)
        
        ax1.set_ylabel("Probability", fontsize = fontsize_fig) 
        if cycle_unit == 'ug/l': 
            ax1.set_xlabel("True Concentration (μg/l)", fontsize = fontsize_fig)
        else:
            ax1.set_xlabel("True Concentration (%s)"%(cycle_unit), fontsize = fontsize_fig)

        plt.yticks([])
        plt.xlim([plot_minimum, plot_maximum])
        for label in ax1.xaxis.get_ticklabels():
            label.set_fontsize(fontsize_fig)
        for label in ax1.yaxis.get_ticklabels():
            label.set_fontsize(fontsize_fig)
        plt.draw()
        plt.pause(2)
   


    return query, target_unit, cycle_unit, plot_maximum, plot_minimum, con_list, con_select_prob_hour, hour_distribution_list





def level_define_date(args, drug_input, covariates_input, dosage_date, dosage_duration, dosage_adjustment_duration, con_pair, con_unit, sample_date, threshold_range, threshold_label, date_input):
    with open(args.patient_example_path, 'r') as file:
        content = file.read()


    soup = BeautifulSoup(content, 'xml')
   

    if soup.query:
        try:
            query = Q.Query(soup)
        
        except Exception as e:
            print('Can not import the following query file :' + args.patient_example_path)
            print(e)
    

    for i in range(len(query.covariates)):
        # print(query.covariates[i].date)
        query.covariates[i].date = str_to_datetime(query.covariates[i].date)
    # query.date = query.date.strftime("%Y-%m-%d %H:%M:%S")
    query.date = date_input
    
    # # =============================================================================
    # # 根据target设置各区间范围
    # target_inefficacyAlarm = float(query.drugs[0].targets[0].inefficacyAlarm)
    # target_min = float(query.drugs[0].targets[0].min)
    # target_best = float(query.drugs[0].targets[0].best)
    # target_max = float(query.drugs[0].targets[0].max)
    # target_toxicityAlarm = float(query.drugs[0].targets[0].toxicityAlarm)
    # target_unit = query.drugs[0].targets[0].unit

    # threshold_range = [[0, target_inefficacyAlarm], [target_inefficacyAlarm, target_min], [target_min, target_max], [target_max, target_toxicityAlarm], [target_toxicityAlarm, float('inf')]]
    # threshold_label = [-2, -1, 0, 1, 2]
    
    
    
    # # print(threshold_range)
    # # print(target_unit)

    # # 修改一下文件中的target
    # query.drugs[0].targets[0].min = float('750')
    # query.drugs[0].targets[0].max = float('1500')
    # query.drugs[0].targets[0].best = float('1000')
    # query.drugs[0].targets[0].unit = 'ug/l'
    # query.drugs[0].targets[0].inefficacyAlarm = float('500')
    # query.drugs[0].targets[0].toxicityAlarm = float('1750')

    # # target_min = float('750')
    # # target_max = float('1500')
    # # target_best = float('1000')
    # # target_unit = 'ug/l'
    # # threshold_range = [[0, target_min], [target_min, target_max], [target_max, float('inf')]]
    # # threshold_label = [-1, 0, 1]
    # # print(threshold_range)
    # # print(target_unit)

    # target_inefficacyAlarm = float(query.drugs[0].targets[0].inefficacyAlarm)
    # target_min = float(query.drugs[0].targets[0].min)
    # target_best = float(query.drugs[0].targets[0].best)
    # target_max = float(query.drugs[0].targets[0].max)
    # target_toxicityAlarm = float(query.drugs[0].targets[0].toxicityAlarm)
    # target_unit = query.drugs[0].targets[0].unit

    # threshold_range = [[0, target_inefficacyAlarm], [target_inefficacyAlarm, target_min], [target_min, target_max], [target_max, target_toxicityAlarm], [target_toxicityAlarm, float('inf')]]
    # threshold_label = [-2, -1, 0, 1, 2]

    # # threshold_range = [[0, target_min], [target_min, target_max], [target_max, float('inf')]]
    # threshold_label = [-1, 0, 1]

    # print(threshold_range)
    # print(target_unit)

    # =============================================================================
    # 改变patient的covariate
    # for i in range(len(query.covariates)):
    #     print(query.covariates[i].covariateId)
    
    # # 这个地方可以加入新的covariate
    # query.covariates = []
    # for cov_index in range(len(covariates_input)):
    #     query.covariates.append(covariates_input[cov_index])

    # # 增加新的covariate
    # covariateId = 'birthdate'
    # date = str_to_datetime(date_input)
    # value = '1960-01-01T00:00:00'
    # unit = '-'
    # datatype = 'date'
    # nature = 'discrete'
    # covariate_new = Q.Covariate.create_covariate(covariateId, date, value, unit, datatype, nature)
    # query.covariates.append(covariate_new)

    # covariateId = 'bodyweight'
    # date = str_to_datetime(date_input)
    # value = '70'
    # unit = 'kg'
    # datatype = 'double'
    # nature = 'continuous'
    # covariate_new = Q.Covariate.create_covariate(covariateId, date, value, unit, datatype, nature)
    # query.covariates.append(covariate_new)

    # covariateId = 'gist'
    # date = str_to_datetime(date_input)
    # value = 'False'
    # unit = '-'
    # datatype = 'bool'
    # nature = 'categorical'
    # covariate_new = Q.Covariate.create_covariate(covariateId, date, value, unit, datatype, nature)
    # query.covariates.append(covariate_new)

    # # Male:1, Female:0
    # covariateId = 'sex'
    # date = str_to_datetime(date_input)
    # value = '1'
    # unit = '-'
    # datatype = 'double'
    # nature = 'continuous'
    # covariate_new = Q.Covariate.create_covariate(covariateId, date, value, unit, datatype, nature)
    # query.covariates.append(covariate_new)

    # =============================================================================
    # 更改药物基本信息
    query.drugs[0].drugId = drug_input.drugId
    query.drugs[0].activePrinciple = drug_input.activePrinciple
    query.drugs[0].brandName = drug_input.brandName
    query.drugs[0].atc = drug_input.atc
    
    # =============================================================================
    # 更改dosage history
    # 如果patient文件中的dosage可用的话，可不修改原有dosage直接使用原始文件
    # 直接添加新的dosage adaptation

    query.drugs[0].dosageHistory.dosageTimeRanges = []
    dosage_history = []
    
    # dosage_start_date = str_to_datetime('2024-03-01T08:00:00')
    # dosage_end_date = str_to_datetime('2024-03-20T08:00:00')
    dosage_start_date = dosage_date
    dosage_end_date = dosage_date + dosage_duration
    lasting_dosage = Q.LastingDosage()
    # lasting_dosage.interval = str_to_time('24:00:00')
    # lasting_dosage.dose.infusionTimeInMinutes = str_to_time('1:00:00')
    # lasting_dosage.dose.unit = 'mg'
    # lasting_dosage.dose.value = float(400)
    # lasting_dosage.formulationAndRoute.absorptionModel = 'extravascular'
    # lasting_dosage.formulationAndRoute.administrationRoute = 'oral'
    # lasting_dosage.formulationAndRoute.administrationName = 'foo bar'
    # lasting_dosage.formulationAndRoute.formulation = 'parenteralSolution'
    lasting_dosage.interval = drug_input.default_dose_interval
    lasting_dosage.dose.infusionTimeInMinutes = drug_input.default_dose.infusionTimeInMinutes
    lasting_dosage.dose.unit = drug_input.default_dose.unit
    lasting_dosage.dose.value = drug_input.default_dose.value
    lasting_dosage.formulationAndRoute.absorptionModel = drug_input.default_formulationAndRoutes.absorptionModel
    lasting_dosage.formulationAndRoute.administrationRoute = drug_input.default_formulationAndRoutes.administrationRoute
    lasting_dosage.formulationAndRoute.administrationName = drug_input.default_formulationAndRoutes.administrationName
    lasting_dosage.formulationAndRoute.formulation = drug_input.default_formulationAndRoutes.formulation
    dosage_history_1 = Q.DosageTime.create_dosage_time_range(dosage_start_date, lasting_dosage, dosage_end_date)
    dosage_history.append(dosage_history_1)

    # print(query.drugs[0].dosageHistory.dosageTimeRanges[0].dosage.dose.infusionTimeInMinutes)
    # query.drugs[0].dosageHistory.dosageTimeRanges[0].dosage.dose.infusionTimeInMinutes = query.drugs[0].dosageHistory.dosageTimeRanges[0].dosage.dose.infusionTimeInMinutes.total_seconds() / 60
    # print(query.drugs[0].dosageHistory.dosageTimeRanges[0].dosage.dose.infusionTimeInMinutes)
    

    query.drugs[0].dosageHistory.dosageTimeRanges = dosage_history
    for i in range(len(query.drugs[0].dosageHistory.dosageTimeRanges)):
        query.drugs[0].dosageHistory.dosageTimeRanges[i].dosage.dose.infusionTimeInMinutes = query.drugs[0].dosageHistory.dosageTimeRanges[i].dosage.dose.infusionTimeInMinutes.total_seconds() / 60
    # =============================================================================
    # 添加sample
    query.drugs[0].samples = []
    analyte_id = drug_input.drugId


    # sample_id = '123456'
    # sampledate = str_to_datetime('2018-07-07T06:00:00')
    # concentration = 0.7
    # sample_unit = 'mg/l'
    # sample1 = Q.Sample.create_sample(sample_id, sampledate, analyte_id, concentration, unit)
    # query.drugs[0].samples.append(sample1)

    last_sample_id = 'last'
    last_sampledate = sample_date
    last_concentration = con_pair[0]
    # sample_unit = 'ug/l'
    sample_unit = con_unit
    last_sample = Q.Sample.create_sample(last_sample_id, last_sampledate, analyte_id, last_concentration, sample_unit)
    query.drugs[0].samples.append(last_sample)

    # =============================================================================
    # 添加request
    query.requests = []
    # nb_points_per_hour = 20

    request1 = R.Request()
    computing_option = R.ComputingOption(R.ParametersTypeEnum.aposteriori, R.CompartmentOptionEnum.allActiveMoieties, True, True, True)
    # dosage_start_date = str_to_datetime('2018-07-06T08:00:00')
    # dosage_end_date = str_to_datetime('2018-07-08T08:00:00')
    request1.requestId = 'aposteriori'
    request1.drugId = drug_input.drugId
    request1.drugModelId = drug_input.drugModelId
    request1.computingTraits = R.PredictionTraits.create_prediction_traits(args.nb_points_per_hour, dosage_start_date, dosage_end_date, computing_option)
    query.requests.append(request1)

    # request2 = R.Request()
    # request2.requestId = 'apriori_1_percentiles'
    # request2.drugId = drugname
    # request2.drugModelId = drugmodel
    # computing_option = R.ComputingOption(R.ParametersTypeEnum.apriori, R.CompartmentOptionEnum.allActiveMoieties, True, True, True)
    # ranks = [5, 10, 25, 50, 75, 90, 95]
    # request2.computingTraits = R.PercentilesTraits.create_percentiles_traits(nb_points_per_hour, dosage_start_date, dosage_end_date, computing_option, ranks)
    # query.requests.append(request2)

    request3 = R.Request()
    request3.requestId = 'dosage_adjustment'
    request3.drugId = drug_input.drugId
    request3.drugModelId = drug_input.drugModelId
    # dosage_adjustment_date = str_to_datetime('2024-03-11T08:00:00')
    # dosage_adjustment_end = str_to_datetime('2024-03-16T08:00:00')
    dosage_adjustment_date = dosage_end_date
    dosage_adjustment_end = dosage_adjustment_date + dosage_adjustment_duration
    computing_option = R.ComputingOption(R.ParametersTypeEnum.aposteriori, R.CompartmentOptionEnum.allActiveMoieties, True, True, True)
    # requests_options = R.AdjustementOptions(R.BestCandidatesOption.allDosages, R.LoadingOption.noLoadingDose, R.RestPeriodOption.noRestPeriod, R.SteadyStateTargetOption.atSteadyState, R.TargetExtractionOption.populationValues, R.FormulationAndRouteSelectionOption.lastFormulationAndRoute)
    requests_options = R.AdjustementOptions(R.BestCandidatesOption.bestDosage, R.LoadingOption.noLoadingDose, R.RestPeriodOption.noRestPeriod, R.SteadyStateTargetOption.atSteadyState, R.TargetExtractionOption.populationValues, R.FormulationAndRouteSelectionOption.lastFormulationAndRoute)
    # requests_options = R.AdjustementOptions(R.BestCandidatesOption.bestDosage, R.LoadingOption.noLoadingDose, R.RestPeriodOption.noRestPeriod, R.SteadyStateTargetOption.atSteadyState, R.TargetExtractionOption.individualTargetsIfDefinitionExistsAndDefinitionIfNoIndividualTarget, R.FormulationAndRouteSelectionOption.lastFormulationAndRoute)
    request3.computingTraits = R.AdjustementTraits.create_adjustements_traits(computing_option, args.nb_points_per_hour, dosage_adjustment_date, dosage_adjustment_end, dosage_adjustment_date, requests_options)                      
    query.requests.append(request3)


    # =============================================================================

    exporter = ExportTqf()
    new_content = exporter.export_to_string(query, args.query_example_path)
  
    # 转换new_content为xml文件
    # root_element = ET.fromstring(new_content)
    # tree = ET.ElementTree(root_element)
    # tree.write("new_content.xml")
    
    
    # print("Tqf Content is:")
    # print(new_content)


    # 转化为object
    results = module.compute_tqf2object(new_content, [args.drug_folder_path])

    # print(results.query_status)
    assert results.query_status == module.QueryStatus.ok
    for srd in results.responses:
        assert srd.computing_response.computing_status == module.ComputingStatus.ok
    
   

    # # 转化为字符串
    results_xml = module.compute_tqf(new_content, [args.drug_folder_path])
    
    



    # 整理出所有cycle data
    cycle_data = results.responses[0].computing_response.data.cycle_data
    cycle_data_list_1 = []
    time_list_1 = []
    index_1 = 0
    for cycle in cycle_data:
        cycle_data_list_1 += cycle.concentrations[0]
        time_list_1 += [index_1 + i for i in range(len(cycle.concentrations[0]))]
        index_1 += len(cycle.concentrations[0])

    

    # =============================================================================
    results_soup = BeautifulSoup(results_xml, 'xml')
    # if results_soup.query:
    #     try:
    #         results_query = Q.Query(results_soup)
        
    #     except Exception as e:
    #         print('Can not import the following query file :' + FILE_NAME)
    #         print(e)
        

    # 然后改变sample，加入新的dosage，算曲线
    # (1) 改变sample
    # print(len(query.drugs[0].samples))
    query.drugs[0].samples = query.drugs[0].samples[:-1]
    # print(len(query.drugs[0].samples))

    last_concentration = con_pair[1]
    last_sample = Q.Sample.create_sample(last_sample_id, last_sampledate, analyte_id, last_concentration, sample_unit)
    query.drugs[0].samples.append(last_sample)
    # print(len(query.drugs[0].samples))



    # (2) 加入新的dosage
    if results_soup.find('score'):
        # 有dosage adjustment

        new_dosage_adjustment = results_soup.find('dosage')
        new_lasting_dosage = Q.LastingDosage()
        new_lasting_dosage.interval = str_to_time(new_dosage_adjustment.find('interval').text)
        infusionTimeInMinutes = new_dosage_adjustment.find('infusionTimeInMinutes').text
        hours = int(float(infusionTimeInMinutes)) // 60
        remaining_minutes = int(float(infusionTimeInMinutes)) % 60
        formatted_time = '{:02d}:{:02d}:00'.format(hours, remaining_minutes)
        new_lasting_dosage.dose.infusionTimeInMinutes = str_to_time(formatted_time)
        new_lasting_dosage.dose.unit = new_dosage_adjustment.find('unit').text
        new_lasting_dosage.dose.value = float(new_dosage_adjustment.find('value').text)
        new_lasting_dosage.formulationAndRoute.absorptionModel = new_dosage_adjustment.find('absorptionModel').text
        new_lasting_dosage.formulationAndRoute.administrationRoute = new_dosage_adjustment.find('administrationRoute').text
        new_lasting_dosage.formulationAndRoute.administrationName = new_dosage_adjustment.find('administrationName').text
        new_lasting_dosage.formulationAndRoute.formulation = new_dosage_adjustment.find('formulation').text
        new_dosage_history = Q.DosageTime.create_dosage_time_range(dosage_adjustment_date, new_lasting_dosage, dosage_adjustment_end)
        dosage_history.append(new_dosage_history)

        query.drugs[0].dosageHistory.dosageTimeRanges = dosage_history
        for i in range(len(query.drugs[0].dosageHistory.dosageTimeRanges)):
            if type(query.drugs[0].dosageHistory.dosageTimeRanges[i].dosage.dose.infusionTimeInMinutes) != float:
                query.drugs[0].dosageHistory.dosageTimeRanges[i].dosage.dose.infusionTimeInMinutes = query.drugs[0].dosageHistory.dosageTimeRanges[i].dosage.dose.infusionTimeInMinutes.total_seconds() / 60

        # (3) 原有dosage算aposteriori
                
        query.requests = []

        # request1 = R.Request()
        # computing_option = R.ComputingOption(R.ParametersTypeEnum.aposteriori, R.CompartmentOptionEnum.allActiveMoieties, True, True, True)
        # # dosage_start_date = str_to_datetime('2018-07-06T08:00:00')
        # # dosage_end_date = str_to_datetime('2018-07-08T08:00:00')
        # request1.requestId = 'aposteriori'
        # request1.drugId = drugname
        # request1.drugModelId = drugmodel
        # request1.computingTraits = R.PredictionTraits.create_prediction_traits(nb_points_per_hour, dosage_start_date, dosage_end_date, computing_option)
        # query.requests.append(request1)

        # 新dosage算prediction
        request2 = R.Request()
        computing_option = R.ComputingOption(R.ParametersTypeEnum.aposteriori, R.CompartmentOptionEnum.allActiveMoieties, True, True, True)
        # dosage_start_date = str_to_datetime('2018-07-06T08:00:00')
        # dosage_end_date = str_to_datetime('2018-07-08T08:00:00')
        request2.requestId = 'prediction'
        request2.drugId = drug_input.drugId
        request2.drugModelId = drug_input.drugModelId
        # request2.computingTraits = R.PredictionTraits.create_prediction_traits(nb_points_per_hour, dosage_adjustment_date, dosage_adjustment_end, computing_option)
        request2.computingTraits = R.PredictionTraits.create_prediction_traits(args.nb_points_per_hour, dosage_start_date, dosage_adjustment_end, computing_option)
        query.requests.append(request2)
        
        
        # =============================================================================

        # print("New dosage adaption:")
        exporter = ExportTqf()
        new_content = exporter.export_to_string(query, args.query_example_path)
        
        # print("Tqf Content is:")
        # print(new_content)


        # 转化为object
        results = module.compute_tqf2object(new_content, [args.drug_folder_path])

        # print(results.query_status)
        assert results.query_status == module.QueryStatus.ok
        for srd in results.responses:
            assert srd.computing_response.computing_status == module.ComputingStatus.ok

        # # 输出具体内容
        # print(f"Results are : {results}")
        # display_computing_query_response(results)
      

        # print("Press any key to continue...")
        # input()

        # # 转化为字符串
        results_xml = module.compute_tqf(new_content, [args.drug_folder_path])
        # print(f"Results are: {results_xml}")
        # sgjsgfjshgf
        # 应该从这个地方提取statistics

        # =============================================================================
        # 评估这个点在哪个区

        statistics_dict = {
            # "startdate": [],
            "mean": [],
            "auc": [],
            "auc24": [],
            "cumulativeAuc": [],
            "residual": [],
            "peak": []
        }

        statistics_soup = BeautifulSoup(results_xml, 'xml')

        cycleDatas = statistics_soup.responses.response.dataPrediction.cycleDatas
        for cycle in cycleDatas.find_all('cycleData'):
            # statistics_dict['startdate'].append(cycle.start.string)
            statistics_dict['mean'].append(float(cycle.statistics.mean.string))
            statistics_dict['auc'].append(float(cycle.statistics.auc.string))
            statistics_dict['auc24'].append(float(cycle.statistics.auc24.string))
            statistics_dict['cumulativeAuc'].append(float(cycle.statistics.cumulativeAuc.string))
            statistics_dict['residual'].append(float(cycle.statistics.residual.string))
            statistics_dict['peak'].append(float(cycle.statistics.peak.string))

        cycle_unit = cycle.unit.string
        target_unit = drug_input.targets[0].unit
        target_max = drug_input.targets[0].max
        target_min = drug_input.targets[0].min
        target_metric = drug_input.targets[0].targetType

        # -2: 因为如果不是完整的周期，-1得到的结果不准确
        # 但会对cumulativeAuc产生影响
        # cumulativeAuc选择-1
        if target_metric == 'auc24DividedByMic':
            # metric_value = statistics_dict['auc'][-2] / drug_input.targets[0].mic

            metric_value = statistics_dict['auc24'][-2] / drug_input.targets[0].mic
            # print(metric_value)
            if drug_input.targets[0].mic_unit[0:2] == 'mg' and cycle_unit[0:2] == 'ug':
                metric_value /= 1000    
        elif target_metric == 'residualDividedByMic':
            metric_value = statistics_dict['residual'][-2] / drug_input.targets[0].mic
            if drug_input.targets[0].mic_unit[0:2] == 'mg' and cycle_unit[0:2] == 'ug':
                metric_value /= 1000    
        elif target_metric == 'cumulativeAuc':
            metric_value = statistics_dict[target_metric][-1]
            if target_unit[0:2] == 'mg' and cycle_unit[0:2] == 'ug':
                metric_value /= 1000
        else:
            metric_value = statistics_dict[target_metric][-2]
            if target_unit[0:2] == 'mg' and cycle_unit[0:2] == 'ug':
                metric_value /= 1000
            
        
        # print(threshold_range)
        for i, interval in enumerate(threshold_range):
            # 判断值在哪个区间
            if interval[0] <= metric_value < interval[1]:
                output_label = threshold_label[i]
                # print(output_label)
                # print(metric_value)
                break

        
        cycle_data = results.responses[0].computing_response.data.cycle_data
        cycle_data_list_2 = []
        time_list_2 = []
        index_2 = 0
        num_cycle = 0
        for cycle in cycle_data:
            num_cycle += 1
            cycle_data_list_2 += cycle.concentrations[0]
            time_list_2 += [index_2 + i for i in range(len(cycle.concentrations[0]))]
            index_2 += len(cycle.concentrations[0])
        # print(num_cycle)

        

        if not args.plot_full_matrix:
            print('=' * 50)
            print(f'Measured: {con_pair[0]}, True: {con_pair[1]}')
            print(f'Output label: {output_label}')

            fig = plt.figure(figsize = (30, 10))
            ax1 = fig.add_subplot(111)
            pllist = []
            lgtextlist = []
            num_start = 0
            
            p1, = plt.plot(time_list_1, cycle_data_list_1, label = 'Old Dosage')
            p1, = plt.plot(time_list_2, cycle_data_list_2, label = 'New Dosage')

            if target_metric == 'residual' or target_metric == 'peak':
                if target_unit[0:2] == 'mg' and cycle_unit[0:2] == 'ug':
                    p1, = plt.plot([target_min * 1000 for _ in range(len(time_list_2))], label = 'Target Min')
                    p1, = plt.plot([target_max * 1000 for _ in range(len(time_list_2))], label = 'Target Max')
                else:
                    p1, = plt.plot([target_min for _ in range(len(time_list_2))], label = 'Target Min')
                    p1, = plt.plot([target_max for _ in range(len(time_list_2))], label = 'Target Max')

            ax1.set_ylabel("Concentration (%s)"%(cycle_unit), fontsize = fontsize_fig)    
            ax1.set_xlabel("Time", fontsize = fontsize_fig)
            plt.title('Concentration - Time Curve' , fontsize = fontsize_fig)
            plt.legend(fontsize = fontsize_fig, loc = 2)

            for label in ax1.xaxis.get_ticklabels():
                label.set_fontsize(fontsize_fig)
            for label in ax1.yaxis.get_ticklabels():
                label.set_fontsize(fontsize_fig)
            plt.draw()
            plt.pause(2)

            


        return True, output_label, metric_value
    else:
        # 没有新的dosage adjustment
        return False, 0, 0


def threeD_plot(args, drug_result_folder):

    if args.plot_3d:

        pattern = re.compile(rf"{re.escape(drug_result_folder)}_(\d+)\.pickle")
        matched_files = []

        # 遍历目录
        for root, _, files in os.walk(basepath + '/' + drug_result_folder + '/'):
            for file in files:
                match = pattern.fullmatch(file)
                if match:
                    number = int(match.group(1))
                    full_path = os.path.join(root, file)
                    matched_files.append((number, full_path))

        # 按提取的数字排序
        matched_files.sort(key=lambda x: x[0])
        print(matched_files)

        target_label = [0, 1, 2, 3, 4]
        target_color_list = ['purple', 'blue', 'green', 'darkorange', 'red']

        hour_index_all = []
        output_label_list_all = []
        output_label_3D = []
        output_postion_3D = []
        for _, file_path in matched_files:
            list_file = open(file_path, 'rb')
            (hour_index, output_label_list, output_value_list, output_con_list, threshold_range, target_color_list) = pickle.load(list_file)
            
            if np.min(np.array(output_con_list)) > 1000:
                output_con_list = (np.array(output_con_list) / 1000).tolist()
                con_unit = '(mg/l)'
            else:
                con_unit = '(μg/l)' 


            hour_index_all.append(hour_index)
            output_label_list_all.append(output_label_list)
            # output_postion_3D += [con + [hour_index] for con in output_con_list]
            output_postion_3D += [[con[1], con[0]] + [hour_index] for con in output_con_list]
            output_label_3D += output_label_list
        

        points = np.array(output_postion_3D)
        labels = np.array(output_label_3D)


        print(hour_index_all)
        print(np.array(output_label_list).shape)
        print(output_con_list)
        print(len(output_label_3D))
        print(len(output_postion_3D))




        plot_minimum = output_con_list[0][0]
        plot_maximum = output_con_list[-1][0]
        print(plot_minimum)
        print(plot_maximum)
        print(np.max(labels))
        print(np.min(labels))

        import plotly.express as px
        print(labels.astype(str))

        
        import plotly.graph_objects as go

        target_label = [i for i in range(len(threshold_range))]
        color_all = ['purple', 'blue', 'green', 'darkorange', 'red']
        color_input = [color_all[label] for label in labels]

        fig = go.Figure(data=[go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=color_input,
                # opacity=0.2,
                opacity=0.1
            ),
            name='',              # 👈 不显示名字
            showlegend=False      # ✅ 禁止出现在图例中
        )])

        # 添加图例项
        color_labels = {
            'purple': 'Inefficacy alarm range',
            'blue': 'Subtherapeutic range',
            'green': 'Therapeutic target range',
            'darkorange': 'Supratherapeutic range ',
            'red': 'Toxicity alarm range'
        }

        label_set = [color_all[index] for index in set(output_label_3D)]
        for color in label_set:
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None],  # 不绘制数据点，只用于图例
                mode='markers',
                marker=dict(size=6, color=color),
                name=color_labels[color]
            ))



        fig.update_layout(
            title=dict(
                text='3D-ETODA',
                font=dict(family='Times New Roman', size= 22),
                x=0.5,  # 居中（0=左，1=右）
                y=0.84  # 向下靠近图（默认是 0.95~1.0）
            ),
            # margin=dict(t=50),  # 减少顶部留白
            margin=dict(l=10, r=10, b=5, t=50),
            width= 1000,
            height= 800,
            scene=dict(
                xaxis=dict(title='True Concentration '+ con_unit, title_font=dict(family='Times New Roman', size = 20)),
                yaxis=dict(title='Measured Concentration '+ con_unit, title_font=dict(family='Times New Roman', size = 20)),
                zaxis=dict(title='Sampling Time (h)', title_font=dict(family='Times New Roman', size = 20)),
                aspectmode='cube',
                camera = dict(eye = dict(x=-1.5, y=-1.5, z=1.5))

            ),
            font=dict(family='Times New Roman', size = 15),
            # title=dict(text='3D-ETODA', font=dict(family='Times New Roman', size=20))
            legend=dict(
                x=0.8,
                y=0.7,
                xanchor='left',
                yanchor='top',
                font=dict(family='Times New Roman', size=19),
            )
        )

        fig.show()




def main():

    parser = parse_the_args()
    args, unknown = parser.parse_known_args()

    drugmodel_file = args.drug_folder_path + args.drug_model_path

    with open(drugmodel_file, 'r') as file:
        content =  file.read()
    soup = BeautifulSoup(content, 'xml')

    new_drug = DrugInformation(soup)

   

    default_coveriates = Default_covariates(soup, args.date_input)
    date_input = args.date_input
    dosage_date = str_to_datetime(args.dosage_date)
    dosage_duration = timedelta(days = int(args.dosage_duration))
    dosage_adjustment_duration = timedelta(days = int(args.dosage_adjustment_duration))
    sample_date = str_to_datetime(args.sample_date)

    input_covariates = []
    id_covariates = []
    value_covariates = []


    for i in range(len(default_coveriates.coveriates)):
        input_covariates.append(default_coveriates.coveriates[i])
        id_covariates.append(default_coveriates.coveriates[i].covariateId)
        value_covariates.append(default_coveriates.coveriates[i].value)
    # =====================================================================


    with open(args.patient_example_path, 'r') as file:
        content = file.read()
    soup = BeautifulSoup(content, 'xml')
   
    if soup.query:
        try:
            query = Q.Query(soup)
            query.drugs[0].drugId = new_drug.drugId

        
        except Exception as e:
            print('Can not import the following query file :' + args.patient_example_path)
            print(e)
    
            
    for i in range(len(query.covariates)):
        query.covariates[i].date = str_to_datetime(query.covariates[i].date)
    query.date = date_input

    query.covariates = []
    for cov_index in range(len(input_covariates)):
        query.covariates.append(input_covariates[cov_index])

    query, target_unit, con_unit, plot_maximum, plot_minimum, con_list, con_prob_hour_list, hour_distribution_list = distribution_plot_priori(args, new_drug, query, dosage_date, dosage_duration, args.percentiles)

    
    threshold_range, target_color_list = fixed_target_sequence_produce(new_drug)

    target_label = [i for i in range(len(threshold_range))]

   

    print('Plot_minimum:  %.4f' %(plot_minimum) + ' %s'%(con_unit))
    print('Plot_maximum:  %.4f' %(plot_maximum) + ' %s'%(con_unit))
    print('Interval: %.4f' % (con_list[1] - con_list[0]) + ' %s'%(con_unit))
    print("=" * 50)

    
    drug_result_folder = new_drug.drugId + "_" + new_drug.targets[0].targetType
    if not os.path.exists(drug_result_folder):
        os.makedirs(drug_result_folder)



    if args.measurement_error:
        
        
        if args.plot_full_matrix:
            con_pair_list = []
            con_pair_index_list = []
            x_matrix = np.zeros((len(con_list), len(con_list)))
            y_matrix = np.zeros((len(con_list), len(con_list)))
            for i in range(len(con_list)):
                for j in range(len(con_list)):
                    y_matrix[i, j] = con_list[i]
                    x_matrix[j, i] = con_list[i]
                    con_pair_list.append([con_list[i], con_list[j]])
                    con_pair_index_list.append([i, j])
        else:
            con_pair_list = []
            con_pair_index_list = []
            mea_list = list(np.linspace(args.mea_range[0], args.mea_range[1], 10))
            true_list = list(np.linspace(args.true_range[0], args.true_range[1], 10))
            for i in range(len(mea_list)):
                for j in range(len(true_list)):
                    con_pair_list.append([mea_list[i], true_list[j]])
                    con_pair_index_list.append([i, j])

        

        output_label_list_all = []
        output_label_matrix_all = []
        # threshold_list_all = []
        output_con_list_all = []
        output_value_list_all = []

        noadjustment_con_list_all = []
    
        hour_index_num = 0
        for hour_index in args.hours:
            print('Hours : %d' % (hour_index))
            sample_data_now = sample_date + timedelta(hours = int(hour_index))
            # 构建输入序列
            # [measurement, true]
            output_label_list = []
            output_label_matrix = np.zeros((len(con_list), len(con_list)))

            output_con_list = []
            output_value_list= []
            noadjustment_con_list = []
            threshold_list = [[] for _ in range(len(target_label) - 1)]

            

            with tqdm(total = len(con_pair_list)) as pbar:
                for con_index in range(len(con_pair_list)):
                
                    con = con_pair_list[con_index]
                    con_pos = con_pair_index_list[con_index]

                    flag, label_now, value_now = level_define_date(args, new_drug, input_covariates, dosage_date, dosage_duration, dosage_adjustment_duration, con, con_unit, sample_data_now, threshold_range, target_label, date_input)
                    
                    if flag:
                        output_con_list.append(con)
                        output_label_matrix[con_pos[1], con_pos[0]] = label_now
                        output_label_list.append(label_now)
                        output_value_list.append(value_now)
                        # print(label_now)
                        # print(con)
                    else:
                        noadjustment_con_list.append(con)
                    pbar.update(1)
            
            output_label_list_all.append(output_label_list)
            output_label_matrix_all.append(output_label_matrix)
            output_con_list_all.append(output_con_list)
            output_value_list_all.append(output_value_list)
            noadjustment_con_list_all.append(noadjustment_con_list)


            # 四周
            # 左
            if args.plot_full_matrix:
                i = 0
                for j in range(1, len(con_list) - 1):
                    if output_label_matrix[i, j - 1] != output_label_matrix[i, j]:
                            threshold_x = (x_matrix[i, j - 1] + x_matrix[i , j]) / 2
                            threshold_y = (y_matrix[i, j - 1] + y_matrix[i , j]) / 2
                            if [threshold_x, threshold_y] not in threshold_list[int(min([output_label_matrix[i, j - 1], output_label_matrix[i, j]]) - 1)]:
                                threshold_list[int(min([output_label_matrix[i, j - 1], output_label_matrix[i, j]]) - 1)].append([threshold_x, threshold_y])

                    if output_label_matrix[i, j + 1] != output_label_matrix[i, j]:
                        threshold_x = (x_matrix[i, j + 1] + x_matrix[i , j]) / 2
                        threshold_y = (y_matrix[i, j + 1] + y_matrix[i , j]) / 2
                        if [threshold_x, threshold_y] not in threshold_list[int(min([output_label_matrix[i, j + 1], output_label_matrix[i, j]]) - 1)]:
                            threshold_list[int(min([output_label_matrix[i, j + 1], output_label_matrix[i, j]]) - 1)].append([threshold_x, threshold_y])

                    if output_label_matrix[i + 1, j - 1] != output_label_matrix[i, j]:
                        threshold_x = (x_matrix[i + 1, j - 1] + x_matrix[i , j]) / 2
                        threshold_y = (y_matrix[i + 1, j - 1] + y_matrix[i , j]) / 2
                        if [threshold_x, threshold_y] not in threshold_list[int(min([output_label_matrix[i + 1, j - 1], output_label_matrix[i, j]]) - 1)]:
                            threshold_list[int(min([output_label_matrix[i + 1, j - 1], output_label_matrix[i, j]]) - 1)].append([threshold_x, threshold_y])
                    
                    if output_label_matrix[i + 1, j] != output_label_matrix[i, j]:
                        threshold_x = (x_matrix[i + 1, j] + x_matrix[i , j]) / 2
                        threshold_y = (y_matrix[i + 1, j] + y_matrix[i , j]) / 2
                        if [threshold_x, threshold_y] not in threshold_list[int(min([output_label_matrix[i + 1, j], output_label_matrix[i, j]]) - 1)]:
                            threshold_list[int(min([output_label_matrix[i + 1, j], output_label_matrix[i, j]]) - 1)].append([threshold_x, threshold_y])

                    if output_label_matrix[i + 1, j + 1] != output_label_matrix[i, j]:
                        threshold_x = (x_matrix[i + 1, j + 1] + x_matrix[i , j]) / 2
                        threshold_y = (y_matrix[i + 1, j + 1] + y_matrix[i , j]) / 2
                        if [threshold_x, threshold_y] not in threshold_list[int(min([output_label_matrix[i + 1, j + 1], output_label_matrix[i, j]]) - 1)]:
                            threshold_list[int(min([output_label_matrix[i + 1, j + 1], output_label_matrix[i, j]]) - 1)].append([threshold_x, threshold_y])


                # 右
                i = len(con_list) - 1
                for j in range(1, len(con_list) - 1):
                    if output_label_matrix[i, j - 1] != output_label_matrix[i, j]:
                            threshold_x = (x_matrix[i, j - 1] + x_matrix[i , j]) / 2
                            threshold_y = (y_matrix[i, j - 1] + y_matrix[i , j]) / 2
                            if [threshold_x, threshold_y] not in threshold_list[int(min([output_label_matrix[i, j - 1], output_label_matrix[i, j]]) - 1)]:
                                threshold_list[int(min([output_label_matrix[i, j - 1], output_label_matrix[i, j]]) - 1)].append([threshold_x, threshold_y])

                    if output_label_matrix[i, j + 1] != output_label_matrix[i, j]:
                        threshold_x = (x_matrix[i, j + 1] + x_matrix[i , j]) / 2
                        threshold_y = (y_matrix[i, j + 1] + y_matrix[i , j]) / 2
                        if [threshold_x, threshold_y] not in threshold_list[int(min([output_label_matrix[i, j + 1], output_label_matrix[i, j]]) - 1)]:
                            threshold_list[int(min([output_label_matrix[i, j + 1], output_label_matrix[i, j]]) - 1)].append([threshold_x, threshold_y])
                
                    if output_label_matrix[i - 1, j - 1] != output_label_matrix[i, j]:
                        threshold_x = (x_matrix[i - 1, j - 1] + x_matrix[i , j]) / 2
                        threshold_y = (y_matrix[i - 1, j - 1] + y_matrix[i , j]) / 2
                        if [threshold_x, threshold_y] not in threshold_list[int(min([output_label_matrix[i - 1, j - 1], output_label_matrix[i, j]]) - 1)]:
                            threshold_list[int(min([output_label_matrix[i - 1, j - 1], output_label_matrix[i, j]]) - 1)].append([threshold_x, threshold_y])
                    
                    if output_label_matrix[i - 1, j] != output_label_matrix[i, j]:
                        threshold_x = (x_matrix[i - 1, j] + x_matrix[i , j]) / 2
                        threshold_y = (y_matrix[i - 1, j] + y_matrix[i , j]) / 2
                        if [threshold_x, threshold_y] not in threshold_list[int(min([output_label_matrix[i - 1, j], output_label_matrix[i, j]]) - 1)]:
                            threshold_list[int(min([output_label_matrix[i - 1, j], output_label_matrix[i, j]]) - 1)].append([threshold_x, threshold_y])

                    if output_label_matrix[i - 1, j + 1] != output_label_matrix[i, j]:
                        threshold_x = (x_matrix[i - 1, j + 1] + x_matrix[i , j]) / 2
                        threshold_y = (y_matrix[i - 1, j + 1] + y_matrix[i , j]) / 2
                        if [threshold_x, threshold_y] not in threshold_list[int(min([output_label_matrix[i - 1, j + 1], output_label_matrix[i, j]]) - 1)]:
                            threshold_list[int(min([output_label_matrix[i - 1, j + 1], output_label_matrix[i, j]]) - 1)].append([threshold_x, threshold_y])

                    
                
                # 上
                j = 0
                for i in range(1, len(con_list) - 1):
                    if output_label_matrix[i - 1, j] != output_label_matrix[i, j]:
                        threshold_x = (x_matrix[i - 1, j] + x_matrix[i , j]) / 2
                        threshold_y = (y_matrix[i - 1, j] + y_matrix[i , j]) / 2
                        if [threshold_x, threshold_y] not in threshold_list[int(min([output_label_matrix[i - 1, j], output_label_matrix[i, j]]) - 1)]:
                            threshold_list[int(min([output_label_matrix[i - 1, j], output_label_matrix[i, j]]) - 1)].append([threshold_x, threshold_y])

                    if output_label_matrix[i + 1, j] != output_label_matrix[i, j]:
                        threshold_x = (x_matrix[i + 1, j] + x_matrix[i , j]) / 2
                        threshold_y = (y_matrix[i + 1, j] + y_matrix[i , j]) / 2
                        if [threshold_x, threshold_y] not in threshold_list[int(min([output_label_matrix[i + 1, j], output_label_matrix[i, j]]) - 1)]:
                            threshold_list[int(min([output_label_matrix[i + 1, j], output_label_matrix[i, j]]) - 1)].append([threshold_x, threshold_y])


                    if output_label_matrix[i - 1, j + 1] != output_label_matrix[i, j]:
                        threshold_x = (x_matrix[i - 1, j + 1] + x_matrix[i , j]) / 2
                        threshold_y = (y_matrix[i - 1, j + 1] + y_matrix[i , j]) / 2
                        if [threshold_x, threshold_y] not in threshold_list[int(min([output_label_matrix[i - 1, j + 1], output_label_matrix[i, j]]) - 1)]:
                            threshold_list[int(min([output_label_matrix[i - 1, j + 1], output_label_matrix[i, j]]) - 1)].append([threshold_x, threshold_y])

                    if output_label_matrix[i, j + 1] != output_label_matrix[i, j]:
                        threshold_x = (x_matrix[i, j + 1] + x_matrix[i , j]) / 2
                        threshold_y = (y_matrix[i, j + 1] + y_matrix[i , j]) / 2
                        if [threshold_x, threshold_y] not in threshold_list[int(min([output_label_matrix[i, j + 1], output_label_matrix[i, j]]) - 1)]:
                            threshold_list[int(min([output_label_matrix[i, j + 1], output_label_matrix[i, j]]) - 1)].append([threshold_x, threshold_y])

                    if output_label_matrix[i + 1, j + 1] != output_label_matrix[i, j]:
                        threshold_x = (x_matrix[i + 1, j + 1] + x_matrix[i , j]) / 2
                        threshold_y = (y_matrix[i + 1, j + 1] + y_matrix[i , j]) / 2
                        if [threshold_x, threshold_y] not in threshold_list[int(min([output_label_matrix[i + 1, j + 1], output_label_matrix[i, j]]) - 1)]:
                            threshold_list[int(min([output_label_matrix[i + 1, j + 1], output_label_matrix[i, j]]) - 1)].append([threshold_x, threshold_y])

                

                # 下
                j = len(con_list) - 1
                for i in range(1, len(con_list) - 1):
                    if output_label_matrix[i - 1, j] != output_label_matrix[i, j]:
                        threshold_x = (x_matrix[i - 1, j] + x_matrix[i , j]) / 2
                        threshold_y = (y_matrix[i - 1, j] + y_matrix[i , j]) / 2
                        if [threshold_x, threshold_y] not in threshold_list[int(min([output_label_matrix[i - 1, j], output_label_matrix[i, j]]) - 1)]:
                            threshold_list[int(min([output_label_matrix[i - 1, j], output_label_matrix[i, j]]) - 1)].append([threshold_x, threshold_y])

                    if output_label_matrix[i + 1, j] != output_label_matrix[i, j]:
                        threshold_x = (x_matrix[i + 1, j] + x_matrix[i , j]) / 2
                        threshold_y = (y_matrix[i + 1, j] + y_matrix[i , j]) / 2
                        if [threshold_x, threshold_y] not in threshold_list[int(min([output_label_matrix[i + 1, j], output_label_matrix[i, j]]) - 1)]:
                            threshold_list[int(min([output_label_matrix[i + 1, j], output_label_matrix[i, j]]) - 1)].append([threshold_x, threshold_y])

                    if output_label_matrix[i - 1, j - 1] != output_label_matrix[i, j]:
                        threshold_x = (x_matrix[i - 1, j - 1] + x_matrix[i , j]) / 2
                        threshold_y = (y_matrix[i - 1, j - 1] + y_matrix[i , j]) / 2
                        if [threshold_x, threshold_y] not in threshold_list[int(min([output_label_matrix[i - 1, j - 1], output_label_matrix[i, j]]) - 1)]:
                            threshold_list[int(min([output_label_matrix[i - 1, j - 1], output_label_matrix[i, j]]) - 1)].append([threshold_x, threshold_y])

                    if output_label_matrix[i, j - 1] != output_label_matrix[i, j]:
                        threshold_x = (x_matrix[i, j - 1] + x_matrix[i , j]) / 2
                        threshold_y = (y_matrix[i, j - 1] + y_matrix[i , j]) / 2
                        if [threshold_x, threshold_y] not in threshold_list[int(min([output_label_matrix[i, j - 1], output_label_matrix[i, j]]) - 1)]:
                            threshold_list[int(min([output_label_matrix[i, j - 1], output_label_matrix[i, j]]) - 1)].append([threshold_x, threshold_y])

                    if output_label_matrix[i + 1, j - 1] != output_label_matrix[i, j]:
                        threshold_x = (x_matrix[i + 1, j - 1] + x_matrix[i , j]) / 2
                        threshold_y = (y_matrix[i + 1, j - 1] + y_matrix[i , j]) / 2
                        if [threshold_x, threshold_y] not in threshold_list[int(min([output_label_matrix[i + 1, j - 1], output_label_matrix[i, j]]) - 1)]:
                            threshold_list[int(min([output_label_matrix[i + 1, j - 1], output_label_matrix[i, j]]) - 1)].append([threshold_x, threshold_y])
                        


                # 中间区域
                for i in range(1, len(con_list) - 1):
                    for j in range(1, len(con_list) - 1):
                        if output_label_matrix[i - 1, j - 1] != output_label_matrix[i, j]:
                            threshold_x = (x_matrix[i - 1, j - 1] + x_matrix[i , j]) / 2
                            threshold_y = (y_matrix[i - 1, j - 1] + y_matrix[i , j]) / 2
                            if [threshold_x, threshold_y] not in threshold_list[int(min([output_label_matrix[i - 1, j - 1], output_label_matrix[i, j]]) - 1)]:
                                threshold_list[int(min([output_label_matrix[i - 1, j - 1], output_label_matrix[i, j]]) - 1)].append([threshold_x, threshold_y])
                        
                        if output_label_matrix[i - 1, j] != output_label_matrix[i, j]:
                            threshold_x = (x_matrix[i - 1, j] + x_matrix[i , j]) / 2
                            threshold_y = (y_matrix[i - 1, j] + y_matrix[i , j]) / 2
                            if [threshold_x, threshold_y] not in threshold_list[int(min([output_label_matrix[i - 1, j], output_label_matrix[i, j]]) - 1)]:
                                threshold_list[int(min([output_label_matrix[i - 1, j], output_label_matrix[i, j]]) - 1)].append([threshold_x, threshold_y])

                        if output_label_matrix[i - 1, j + 1] != output_label_matrix[i, j]:
                            threshold_x = (x_matrix[i - 1, j + 1] + x_matrix[i , j]) / 2
                            threshold_y = (y_matrix[i - 1, j + 1] + y_matrix[i , j]) / 2
                            if [threshold_x, threshold_y] not in threshold_list[int(min([output_label_matrix[i - 1, j + 1], output_label_matrix[i, j]]) - 1)]:
                                threshold_list[int(min([output_label_matrix[i - 1, j + 1], output_label_matrix[i, j]]) - 1)].append([threshold_x, threshold_y])

                        if output_label_matrix[i, j - 1] != output_label_matrix[i, j]:
                            threshold_x = (x_matrix[i, j - 1] + x_matrix[i , j]) / 2
                            threshold_y = (y_matrix[i, j - 1] + y_matrix[i , j]) / 2
                            if [threshold_x, threshold_y] not in threshold_list[int(min([output_label_matrix[i, j - 1], output_label_matrix[i, j]]) - 1)]:
                                threshold_list[int(min([output_label_matrix[i, j - 1], output_label_matrix[i, j]]) - 1)].append([threshold_x, threshold_y])


                        if output_label_matrix[i, j + 1] != output_label_matrix[i, j]:
                            threshold_x = (x_matrix[i, j + 1] + x_matrix[i , j]) / 2
                            threshold_y = (y_matrix[i, j + 1] + y_matrix[i , j]) / 2
                            if [threshold_x, threshold_y] not in threshold_list[int(min([output_label_matrix[i, j + 1], output_label_matrix[i, j]]) - 1)]:
                                threshold_list[int(min([output_label_matrix[i, j + 1], output_label_matrix[i, j]]) - 1)].append([threshold_x, threshold_y])

                        if output_label_matrix[i + 1, j - 1] != output_label_matrix[i, j]:
                            threshold_x = (x_matrix[i + 1, j - 1] + x_matrix[i , j]) / 2
                            threshold_y = (y_matrix[i + 1, j - 1] + y_matrix[i , j]) / 2
                            if [threshold_x, threshold_y] not in threshold_list[int(min([output_label_matrix[i + 1, j - 1], output_label_matrix[i, j]]) - 1)]:
                                threshold_list[int(min([output_label_matrix[i + 1, j - 1], output_label_matrix[i, j]]) - 1)].append([threshold_x, threshold_y])
                        
                        if output_label_matrix[i + 1, j] != output_label_matrix[i, j]:
                            threshold_x = (x_matrix[i + 1, j] + x_matrix[i , j]) / 2
                            threshold_y = (y_matrix[i + 1, j] + y_matrix[i , j]) / 2
                            if [threshold_x, threshold_y] not in threshold_list[int(min([output_label_matrix[i + 1, j], output_label_matrix[i, j]]) - 1)]:
                                threshold_list[int(min([output_label_matrix[i + 1, j], output_label_matrix[i, j]]) - 1)].append([threshold_x, threshold_y])

                        if output_label_matrix[i + 1, j + 1] != output_label_matrix[i, j]:
                            threshold_x = (x_matrix[i + 1, j + 1] + x_matrix[i , j]) / 2
                            threshold_y = (y_matrix[i + 1, j + 1] + y_matrix[i , j]) / 2
                            if [threshold_x, threshold_y] not in threshold_list[int(min([output_label_matrix[i + 1, j + 1], output_label_matrix[i, j]]) - 1)]:
                                threshold_list[int(min([output_label_matrix[i + 1, j + 1], output_label_matrix[i, j]]) - 1)].append([threshold_x, threshold_y])




            if args.unit_change_flag:

                if args.unit_change_to == 'mg/l' and con_unit == 'ug/l':


                    fig = plt.figure(figsize = (15, 20))
                    gs = gridspec.GridSpec(2, 1, height_ratios = [3, 1])
                    ax1 = fig.add_subplot(gs[0])
                    pllist = []
                    lgtextlist = []
                    num_start = 0



                    for i in range(len(target_label)):
                        index = np.where(np.array(output_label_list) == target_label[i])
                        x = np.array(output_con_list)[index, 1] / 1000
                        y = np.array(output_con_list)[index, 0] / 1000
                        plt.scatter(x, y, color = target_color_list[i])
                    
                    if len(noadjustment_con_list)!= 0:
                        x_non = np.array(noadjustment_con_list)[:, 1] / 1000
                        y_non = np.array(noadjustment_con_list)[:, 0] / 1000
                        plt.scatter(x_non, y_non, color = 'black')
                    
                    if args.plot_full_matrix:
                        for th in range(len(threshold_list)):
                            if len(threshold_list[th]) != 0 :
                                # print(threshold_list[th])
                                # sorted_threshold_list = np.array(threshold_list[th])[np.lexsort((np.array(threshold_list[th])[:,0], np.array(threshold_list[th])[:,1]))]
                                # plt.plot(sorted_threshold_list[:, 1], sorted_threshold_list[:, 0], linestyle = '--', color = 'black')

                                sorted_threshold_list = np.array(threshold_list[th])[np.lexsort((np.array(threshold_list[th])[:,0], np.array(threshold_list[th])[:,1]))]
                                # plt.plot(sorted_threshold_list[:, 1], sorted_threshold_list[:, 0], linestyle = '--', color = 'black')
                                # plt.plot(sorted_threshold_list[:, 1], sorted_threshold_list[:, 0], marker = '.', color = 'black')
                                # print(sorted_threshold_list[:, 1])
                                # print(sorted_threshold_list[:, 0])
                                x_select = copy.deepcopy(sorted_threshold_list[:, 1]/ 1000)
                                y_select = copy.deepcopy(sorted_threshold_list[:, 0]/ 1000)
                                distances = np.sqrt((x_select - np.mean(x_select)) ** 2 + (y_select - np.mean(y_select)) ** 2)
                                ref_index = np.argmax(distances)
                                x_ref = x_select[ref_index]
                                y_ref = y_select[ref_index]
                                x_select = np.delete(x_select, ref_index)
                                y_select = np.delete(y_select, ref_index)
                                
                                while len(list(x_select)) != 0:
                                    distances = np.sqrt((x_select - x_ref) ** 2 + (y_select - y_ref) ** 2)
                                    ref_index = np.argmin(distances)
                                    plt.plot([x_ref, x_select[ref_index]], [y_ref, y_select[ref_index]], linestyle = '--', color = 'black')
                                    x_ref = x_select[ref_index]
                                    y_ref = y_select[ref_index]
                                    x_select = np.delete(x_select, ref_index)
                                    y_select = np.delete(y_select, ref_index)

                    

                    ax1.set_ylabel("Measured Concentration (mg/l)", fontsize = fontsize_fig)    
                    ax1.set_xlabel("True Concentration (mg/l)", fontsize = fontsize_fig)
                    # plt.title('Error Grid (Sample Date : %s)'% (sample_data_now.strftime("%Y-%m-%d %H:%M:%S")) , fontsize = fontsize_fig)
                    # plt.legend(fontsize =fontsize_fig, loc = 2)
                    # plt.title('Error Grid - AUC/MIC (Sampling time: %d hours)' % (hour_index), fontsize = fontsize_fig)
                    plt.title('Error Grid - Trough (Sampling time: %d hours)' % (hour_index), fontsize = fontsize_fig)


                    plt.xlim([(plot_minimum - (plot_maximum - plot_minimum) / args.num_con)/ 1000, (plot_maximum + (plot_maximum - plot_minimum) / args.num_con)/ 1000])
                    plt.ylim([(plot_minimum - (plot_maximum - plot_minimum) / args.num_con)/ 1000, (plot_maximum + (plot_maximum - plot_minimum) / args.num_con)/ 1000])


                    for label in ax1.xaxis.get_ticklabels():
                        label.set_fontsize(fontsize_fig)
                    for label in ax1.yaxis.get_ticklabels():
                        label.set_fontsize(fontsize_fig)
                    

                    # fig = plt.figure(figsize = (15, 5))
                    ax2 = fig.add_subplot(gs[1])
                    sns.kdeplot([m / 1000 for m in hour_distribution_list[hour_index_num]], shade = True)
                    plt.title('Distribution (Sampling time: %d hours)' % (args.hours[hour_index_num]), fontsize = fontsize_fig)
                    ax2.set_ylabel("Probability", fontsize = fontsize_fig)    
                    ax2.set_xlabel("True Concentration (mg/l)", fontsize = fontsize_fig)

                    plt.yticks([])
                    plt.xlim([(plot_minimum - (plot_maximum - plot_minimum) / args.num_con)/ 1000, (plot_maximum + (plot_maximum - plot_minimum) / args.num_con) / 1000])
                
                    for label in ax2.xaxis.get_ticklabels():
                        label.set_fontsize(fontsize_fig)
                    for label in ax2.yaxis.get_ticklabels():
                        label.set_fontsize(fontsize_fig)

                    plt.tight_layout()
                    plt.draw()
                    plt.savefig(basepath + "/" + drug_result_folder + "/" + drug_result_folder + "_" + str(hour_index) + '.jpg', dpi = 300)
                    plt.pause(2)
            
                
            else:


                fig = plt.figure(figsize = (15, 20))
                gs = gridspec.GridSpec(2, 1, height_ratios = [3, 1])
                ax1 = fig.add_subplot(gs[0])
                pllist = []
                lgtextlist = []
                num_start = 0



                for i in range(len(target_label)):
                    index = np.where(np.array(output_label_list) == target_label[i])
                    x = np.array(output_con_list)[index, 1]
                    y = np.array(output_con_list)[index, 0]
                    plt.scatter(x, y, color = target_color_list[i])
                
                if len(noadjustment_con_list)!= 0:
                    x_non = np.array(noadjustment_con_list)[:, 1]
                    y_non = np.array(noadjustment_con_list)[:, 0]
                    plt.scatter(x_non, y_non, color = 'black')
                
                if args.plot_full_matrix:
                    for th in range(len(threshold_list)):
                        if len(threshold_list[th]) != 0 :
                            
                            sorted_threshold_list = np.array(threshold_list[th])[np.lexsort((np.array(threshold_list[th])[:,0], np.array(threshold_list[th])[:,1]))]
                        
                            x_select = copy.deepcopy(sorted_threshold_list[:, 1])
                            y_select = copy.deepcopy(sorted_threshold_list[:, 0])
                            distances = np.sqrt((x_select - np.mean(x_select)) ** 2 + (y_select - np.mean(y_select)) ** 2)
                            ref_index = np.argmax(distances)
                            x_ref = x_select[ref_index]
                            y_ref = y_select[ref_index]
                            x_select = np.delete(x_select, ref_index)
                            y_select = np.delete(y_select, ref_index)
                            
                            while len(list(x_select)) != 0:
                                distances = np.sqrt((x_select - x_ref) ** 2 + (y_select - y_ref) ** 2)
                                ref_index = np.argmin(distances)
                                plt.plot([x_ref, x_select[ref_index]], [y_ref, y_select[ref_index]], linestyle = '--', color = 'black')
                                x_ref = x_select[ref_index]
                                y_ref = y_select[ref_index]
                                x_select = np.delete(x_select, ref_index)
                                y_select = np.delete(y_select, ref_index)

                
                if con_unit == 'ug/l': 
                    ax1.set_ylabel("Measured Concentration (μg/l)", fontsize = fontsize_fig)    
                    ax1.set_xlabel("True Concentration (μg/l)", fontsize = fontsize_fig)
                else:
                    ax1.set_ylabel("Measured Concentration (%s)"%(con_unit), fontsize = fontsize_fig)    
                    ax1.set_xlabel("True Concentration (%s)"%(con_unit), fontsize = fontsize_fig)
                # plt.title('Error Grid (Sample Date : %s)'% (sample_data_now.strftime("%Y-%m-%d %H:%M:%S")) , fontsize = fontsize_fig)
                # plt.legend(fontsize =fontsize_fig, loc = 2)
                plt.title('Error Grid (Sampling time: %d hours)' % (hour_index), fontsize = fontsize_fig)

                plt.xlim([plot_minimum - (plot_maximum - plot_minimum) / args.num_con, plot_maximum + (plot_maximum - plot_minimum) / args.num_con])
                plt.ylim([plot_minimum - (plot_maximum - plot_minimum) / args.num_con, plot_maximum + (plot_maximum - plot_minimum) / args.num_con])

                # plt.xlim([0, plot_maximum + plot_minimum])
                # plt.ylim([0, plot_maximum + plot_minimum])


                for label in ax1.xaxis.get_ticklabels():
                    label.set_fontsize(fontsize_fig)
                for label in ax1.yaxis.get_ticklabels():
                    label.set_fontsize(fontsize_fig)
                

                # fig = plt.figure(figsize = (15, 5))
                ax2 = fig.add_subplot(gs[1])
                sns.kdeplot(hour_distribution_list[hour_index_num], shade = True)
                plt.title('Distribution (Sampling time: %d hours)' % (args.hours[hour_index_num]), fontsize = fontsize_fig)
                ax2.set_ylabel("Probability", fontsize = fontsize_fig)
                if con_unit == 'ug/l': 
                    ax2.set_xlabel("True Concentration (μg/l)", fontsize = fontsize_fig)
                else:   
                    ax2.set_xlabel("True Concentration (%s)"%(con_unit), fontsize = fontsize_fig)

                plt.yticks([])
                plt.xlim([plot_minimum - (plot_maximum - plot_minimum) / args.num_con, plot_maximum + (plot_maximum - plot_minimum) / args.num_con])

                
            
                for label in ax2.xaxis.get_ticklabels():
                    label.set_fontsize(fontsize_fig)
                for label in ax2.yaxis.get_ticklabels():
                    label.set_fontsize(fontsize_fig)

                plt.tight_layout()
                plt.draw()
                plt.savefig(basepath + "/" + drug_result_folder + "/" + drug_result_folder + "_" + str(hour_index) + '.jpg', dpi = 300)
                plt.pause(2)


            

            print("=" * 50)


            if args.plot_full_matrix:
            
                print('Sampling time after a new administration in steady state: ')
                print('Hours : %d' % (hour_index))
                numbers = [output_label_list.count(label) for label in target_label]
                percentages = [num / len(output_label_list)for num in numbers]
                

                weighted = []
                for num in range(len(con_pair_list)):
                    index_now = con_list.index(con_pair_list[num][1])
                    weighted.append(con_prob_hour_list[hour_index_num][index_now])


                weighted_score = []
                for label_index in target_label:
                    index_list = [ind for ind, val in enumerate(output_label_list) if val == label_index]
                    weighted_score.append(sum([weighted[indind] for indind in index_list])/sum(weighted))


                result = {
                    'Ranges': ['[0, Inefficacy]', '[Inefficacy, Minimum]', '[Minimum, Maximum]', '[Maximum, Toxicity]', '[Toxicity, Infinity]'],
                    'Numbers': numbers,
                    'Percentages': percentages,
                    'Weighted Score': weighted_score
                }
                df = pd.DataFrame(result)
                print(df)

                print("=" * 50)

            hour_index_num += 1


                        
            from pathlib import Path

            def save_pickle(obj, base_path, drug_result_folder, hour_index):
                
                folder = Path(base_path) / drug_result_folder
                folder.mkdir(parents=True, exist_ok=True)  

                # 拼接文件名
                filename = f"{drug_result_folder}_{hour_index}.pickle"
                filepath = folder / filename

                # 保存 pickle 文件
                with open(filepath, "wb") as f:
                    pickle.dump(obj, f)

            


            save_pickle((hour_index, output_label_list, output_value_list, output_con_list, threshold_range, target_color_list), basepath, drug_result_folder, hour_index)   

 


if __name__ == "__main__":
    main()
