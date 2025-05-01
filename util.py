##################################################################################
# This module contains local utility function definitions used in the notebooks.
##################################################################################

import pandas as pd
import matplotlib.pyplot as plt
import math
import os
import ws3.opt
import pickle
import numpy as np
from math import pi



def schedule_harvest_areacontrol(fm, max_harvest=1., period=None, acode='harvest', util=0.85, 
                                 target_masks=None, target_areas=None,
                                 target_scalefactors=None,
                                 mask_area_thresh=0.,
                                 verbose=0):
    """
    Implement a priority queue heuristic harvest scheduler.

    This function can do a bunch of stuff depending on the parameters, but the basic default
    behaviour is to automatically calculate a per-development-type target periodic harvest area
    based on the concept of normal age class distributions and optimal rotation ages.
    """
    if not target_areas:
        if not target_masks: # default to AU-wise THLB 
            au_vals = []
            au_agg = []
            for au in fm.theme_basecodes(5):
                mask = '? 1 ? ? ? %s' % au
                masked_area = fm.inventory(0, mask=mask)
                if masked_area > mask_area_thresh:
                    au_vals.append(au)
                else:
                    au_agg.append(au)
                    if verbose > 0:
                        print('adding to au_agg', mask, masked_area)
            if au_agg:
                fm._themes[5]['areacontrol_au_agg'] = au_agg 
                if fm.inventory(0, mask='? ? ? ? ? areacontrol_au_agg') > mask_area_thresh:
                    au_vals.append('areacontrol_au_agg')
            target_masks = ['? 1 ? ? ? %s' % au for au in au_vals]
        target_areas = []
        for i, mask in enumerate(target_masks): # compute area-weighted mean CMAI age for each masked DT set
            masked_area = fm.inventory(0, mask=mask, verbose=verbose)
            if not masked_area: continue
            r = sum((fm.dtypes[dtk].ycomp('totvol').mai().ytp().lookup(0) * fm.dtypes[dtk].area(0)) for dtk in fm.unmask(mask))
            r /= masked_area
            asf = 1. if not target_scalefactors else target_scalefactors[i]  
            ta = max_harvest * (1/r) * fm.period_length * masked_area * asf
            target_areas.append(ta)
    periods = fm.periods if not period else [period]
    for period in periods:
        for mask, target_area in zip(target_masks, target_areas):
            if verbose > 0:
                print('calling areaselector', period, acode, target_area, mask)
            fm.areaselector.operate(period, acode, target_area, mask=mask, verbose=verbose)
    sch = fm.compile_schedule()
    return sch



################################################
# HWP effect
################################################

def calculate_initial_c_value_stock(fm, i, product_coefficient, util=0.85):
    """
    Calculate carbon stock for harvested wood products for period 1.
    """
    return fm.compile_product(i, f'totvol * {product_coefficient} * {util}')  * 460 * 0.5  / fm.period_length
    

def calculate_c_value_stock(fm, i, product_coefficient, decay_rate, util=0.85):      
    """
    Calculate carbon stock for harvested wood products for period `i`.
    """
    period = math.ceil(i / fm.period_length)
    return (
        sum(fm.compile_product(period, f'totvol * {product_coefficient}  * {util}') / fm.period_length * (1 - decay_rate)**(i - j)
        for j in range(1, i + 1)
        ) * 460 * 0.5 
    )
    

def hwp_carbon_stock(fm, products, product_coefficients, decay_rates, hwp_pool_effect_value):
    """
    Compile periodic harvested wood products carbon stocks data.
    """
    from util import calculate_c_value_stock, calculate_initial_c_value_stock
    data_carbon_stock = {'period': [], 'co2_stock': []}    
    for i in range(0, fm.horizon * 10 + 1):
        period_value = i
        co2_values_stock = []
        for product in products:
            product_coefficient = product_coefficients[product]
            decay_rate = decay_rates[product]            
            if i == 0:
                co2_values_stock.append(0)
            if i == 1:
                co2_values_stock.append(hwp_pool_effect_value * calculate_initial_c_value_stock(fm, i, product_coefficient))
            else:
                co2_values_stock.append(hwp_pool_effect_value * calculate_c_value_stock(fm, i, product_coefficient, decay_rate))
        co2_value_stock = sum(co2_values_stock) / 1000
        data_carbon_stock['period'].append(period_value)
        data_carbon_stock['co2_stock'].append(co2_value_stock)    
    df_carbon_stock = pd.DataFrame(data_carbon_stock)    
    return df_carbon_stock

def calculate_initial_co2_value_emission(fm, i, product_coefficient, decay_rate, util=0.85):
    return fm.compile_product(i, f'totvol * {product_coefficient} * {util}') * 460 * 0.5 * (44 / 12) * decay_rate  / fm.period_length

def calculate_initial_co2_value_emission_residue(fm, i, util=0.85):
    return fm.compile_product(i, f'totvol  * {1-util}') * 460 * 0.5 * 44 / 12  / fm.period_length

def calculate_co2_value_emission(fm, i, product_coefficient, decay_rate, util=0.85):
    period = math.ceil(i / fm.period_length)
    return (
        sum(fm.compile_product(period, f'totvol * {product_coefficient}* {util}') *  (1 - decay_rate)**(i - j) / fm.period_length
        for j in range(1, i + 1)
        ) * 460 * 0.5 * 44 / 12 * decay_rate 
 )

def calculate_co2_value_emission_residue(fm, i, util=0.85):
    period = math.ceil(i / fm.period_length)
    return (
        fm.compile_product(period, f'totvol * {1-util}') * 460 * 0.5 * (44 / 12) / fm.period_length
 )

# Emission (by year)
def hwp_carbon_emission(fm, products, product_coefficients, decay_rates, hwp_pool_effect_value):
    from util import calculate_co2_value_emission, calculate_initial_co2_value_emission, calculate_co2_value_emission_residue, calculate_initial_co2_value_emission_residue
    data_carbon_emission = {'period': [], 'co2_emission': []}    
    for i in range(0, fm.horizon * 10  + 1):
        period_value = i
        co2_values_emission = []        
        for product in products:
            product_coefficient = product_coefficients[product]
            decay_rate = decay_rates[product]            
            if i == 0:
                co2_values_emission.append(0)
            elif i == 1:
                co2_values_emission.append(hwp_pool_effect_value * (calculate_initial_co2_value_emission(fm, i, product_coefficient, decay_rate) + calculate_initial_co2_value_emission_residue(fm, i) ))
            else:
                co2_values_emission.append(hwp_pool_effect_value * (calculate_co2_value_emission(fm, i, product_coefficient, decay_rate) + calculate_co2_value_emission_residue(fm, i)))
        
        
        co2_value_emission = sum(co2_values_emission) / 1000
        data_carbon_emission['period'].append(period_value)
        data_carbon_emission['co2_emission'].append(co2_value_emission)    
    df_carbon_emission = pd.DataFrame(data_carbon_emission)
    return df_carbon_emission


def hwp_carbon_emission_immed(fm):
    data_carbon_emission_immed = {'period': [], 'co2_emission_immed': []}    
    for i in range(0, fm.horizon * 10  + 1):
        period_value = i
        co2_values_emission_immed = []                    
        if i == 0:
            co2_values_emission_immed.append(0)
        else:
            period = math.ceil(i / fm.period_length)
            co2_values_emission_immed.append(fm.compile_product(period, 'totvol') * 460 * 0.5 * (44 / 12) / fm.period_length)
        co2_value_emission_immed = sum(co2_values_emission_immed) / 1000
        data_carbon_emission_immed['period'].append(period_value)
        data_carbon_emission_immed['co2_emission_immed'].append(co2_value_emission_immed)    
    df_carbon_emission_immed = pd.DataFrame(data_carbon_emission_immed)
    return df_carbon_emission_immed

################################################
# Displacement effect
################################################
# Displacement of concrete manufacturing
def calculate_concrete_volume(fm, i, product_coefficients, credibility, clt_conversion_rate, util=0.85):            
    period = math.ceil(i / fm.period_length)
    return fm.compile_product(period,'totvol') * product_coefficients['pclt'] * credibility * util / clt_conversion_rate 


# Iterate through the rows of the DataFrame
def emission_concrete_manu(fm, product_coefficients, credibility, clt_conversion_rate, co2_concrete_manu_factor, displacement_effect):
    from util import  calculate_concrete_volume
    df_emission_concrete_manu = {'period': [], 'co2_concrete_manu': []}
    for i in range(0, fm.horizon *10   + 1 ):
        period_value = i
        co2_concrete_manu = []
        if i == 0:
            co2_concrete_manu = 0
        else:
            concrete_volume = calculate_concrete_volume(fm, i, product_coefficients, credibility, clt_conversion_rate)
            co2_concrete_manu = concrete_volume * co2_concrete_manu_factor / (fm.period_length * 1000)
        df_emission_concrete_manu['period'].append(period_value)
        df_emission_concrete_manu['co2_concrete_manu'].append(co2_concrete_manu)
    # Create a DataFrame from the dictionary
    df_emission_concrete_manu = pd.DataFrame(df_emission_concrete_manu)
    return df_emission_concrete_manu


# Displacement of concrete landfill
def emission_concrete_landfill(fm, product_coefficients, credibility, clt_conversion_rate, co2_concrete_landfill_factor, displacement_effect):
    from util import  calculate_concrete_volume
    df_emission_concrete_landfill = {'period': [], 'co2_concrete_landfill': []}   
    # Iterate through the rows of the DataFrame
    for i in range(0, fm.horizon *10   + 1 ):
        period_value = i
        co2_concrete_landfill = []
        if i == 0:
            co2_concrete_landfill = 0
        else:
            concrete_volume = calculate_concrete_volume(fm, i, product_coefficients, credibility, clt_conversion_rate)
            co2_concrete_landfill = concrete_volume * co2_concrete_landfill_factor  / (fm.period_length * 1000)                       
        df_emission_concrete_landfill['period'].append(period_value)
        df_emission_concrete_landfill['co2_concrete_landfill'].append(co2_concrete_landfill)    
    # Create a DataFrame from the dictionary
    df_emission_concrete_landfill = pd.DataFrame(df_emission_concrete_landfill)
    return df_emission_concrete_landfill
################################################

def compile_scenario(fm, case_study, obj_mode, scenario_name):
    oha = [fm.compile_product(period, '1.', acode='harvest') for period in fm.periods]
    ohv = [fm.compile_product(period, 'totvol * 0.85', acode='harvest') for period in fm.periods]
    ogs = [fm.inventory(period, 'totvol') for period in fm.periods]
    data = {'period':fm.periods, 
            'oha':oha, 
            'ohv':ohv, 
            'ogs':ogs}
    df = pd.DataFrame(data)

    csv_folder_path = os.path.join('./outputs/csv', case_study)
    if not os.path.exists(csv_folder_path):
        os.makedirs(csv_folder_path)
    
    csv_file_path = os.path.join(csv_folder_path, f'{case_study}_{obj_mode}_{scenario_name}_compile_scenario.csv')
    df.to_csv(csv_file_path, index=False)
    
    return df


def plot_scenario(df, case_study, obj_mode, scenario_name):
    import os
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].bar(df.period, df.oha)
    ax[0].set_ylim(0, None)
    ax[0].set_title('Harvested area (ha)')
    ax[1].bar(df.period, df.ohv)
    ax[1].set_ylim(0, None)
    ax[1].set_title('Harvested volume (m3)')
    ax[2].bar(df.period, df.ogs)
    ax[2].set_ylim(0, None)
    ax[2].set_title('Growing Stock (m3)')
    
    plt.tight_layout()
    
    folder_path = os.path.join('./outputs/fig', case_study)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)   
    file_name = f"{case_study}_{obj_mode}_{scenario_name}_scheduling.pdf"
    file_path = os.path.join(folder_path, file_name)
    
    # Save and show plot
    plt.savefig(file_path)
    plt.show()
    plt.close()
    print(f"Plot saved to {file_path}")


    
    return fig, ax


def plot_results(fm):
    pareas = [fm.compile_product(period, '1.') for period in fm.periods]
    pvols = [fm.compile_product(period, 'totvol') for period in fm.periods]
    df = pd.DataFrame({'period':fm.periods, 'ha':pareas, 'hv':pvols})
    fig, ax = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
    ax[0].set_ylabel('harvest area')
    ax[0].bar(df.period, df.ha)
    ax[1].set_ylabel('harvest volume')
    ax[1].bar(df.period, df.hv)
    ax[2].set_ylabel('harvest volume:area ratio')
    ax[2].bar(df.period, (df.hv/df.ha).fillna(0))
    ax[2].set_ylim(0, None)
    return fig, ax, df


def compile_scenario_maxstock(fm, case_study, obj_mode, scenario_name):
    oha = [fm.compile_product(period, '1.', acode='harvest') for period in fm.periods]
    ohv = [fm.compile_product(period, 'totvol * 0.85', acode='harvest') for period in fm.periods]
    ogs = [fm.inventory(period, 'totvol') for period in fm.periods]
    ocp = [fm.inventory(period, 'ecosystem') for period in fm.periods]
    ocf = [fm.inventory(period, 'total_emissions') for period in fm.periods]
    data = {'period':fm.periods, 
            'oha':oha, 
            'ohv':ohv, 
            'ogs':ogs,
            'ocp':ocp,
            'ocf':ocf}
    df = pd.DataFrame(data)


    csv_folder_path = os.path.join('./outputs/csv', case_study)
    if not os.path.exists(csv_folder_path):
        os.makedirs(csv_folder_path)
    
    csv_file_path = os.path.join(csv_folder_path, f'{case_study}_{obj_mode}_{scenario_name}_compile_scenario_maxstock.csv')
    df.to_csv(csv_file_path, index=False)
    
    return df


def plot_scenario_maxstock(df, case_study, obj_mode, scenario_name):
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    # Plot and label the first subplot for harvested area
    ax[0].bar(df.period, df.oha)
    ax[0].set_ylim(0, None)
    ax[0].set_title('Harvested area')
    ax[0].set_xlabel('Period')
    ax[0].set_ylabel('Area (ha)')
    
    # Plot and label the second subplot for harvested volume
    ax[1].bar(df.period, df.ohv)
    ax[1].set_ylim(0, None)
    ax[1].set_title('Harvested volume')
    ax[1].set_xlabel('Period')
    ax[1].set_ylabel('Volume (m3)')

    # Plot and label the third subplot for growing stock
    ax[2].bar(df.period, df.ogs)
    ax[2].set_ylim(0, None)
    ax[2].set_xlabel('Period')
    ax[2].set_title('Growing Stock')
    ax[2].set_ylabel('Volume (m3)')

    # Plot and label the fourth subplot for ecosystem carbon stock
    ax[3].bar(df.period, df.ocp)
    ax[3].set_ylim(0, None)
    ax[3].set_title('Ecosystem C stock')
    ax[3].set_xlabel('Period')
    ax[3].set_ylabel('Stock (ton)')

    # # Plot and label the fifth subplot for total carbon emission
    # ax[4].bar(df.period, df.ocf)
    # ax[4].set_ylim(0, None)
    # ax[4].set_title('Total Carbon Emission')
    # ax[4].set_xlabel('Period')
    # ax[4].set_ylabel('tons of C')

    plt.tight_layout()
    
    folder_path = os.path.join('./outputs/fig', case_study)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)   
    file_name = f"{case_study}_{obj_mode}_{scenario_name}_scheduling_maxstock.pdf"
    file_path = os.path.join(folder_path, file_name)
    
    # Save and show plot
    plt.savefig(file_path)
    plt.show()
    plt.close()
    print(f"Plot saved to {file_path}")
    
    return fig, ax


def compile_scenario_minemission(fm, case_study, obj_mode, scenario_name):
    oha = [fm.compile_product(period, '1.', acode='harvest') for period in fm.periods]
    ohv = [fm.compile_product(period, 'totvol * 0.85', acode='harvest') for period in fm.periods]
    ogs = [fm.inventory(period, 'totvol') for period in fm.periods]
    ocp = [fm.inventory(period, 'ecosystem') for period in fm.periods]
    ocf = [fm.inventory(period, 'net_emission') for period in fm.periods]
    data = {'period':fm.periods, 
            'oha':oha, 
            'ohv':ohv, 
            'ogs':ogs,
            'ocp':ocp,
            'ocf':ocf}
    df = pd.DataFrame(data)

    csv_folder_path = os.path.join('./outputs/csv', case_study)
    if not os.path.exists(csv_folder_path):
        os.makedirs(csv_folder_path)
    
    csv_file_path = os.path.join(csv_folder_path, f'{case_study}_{obj_mode}_{scenario_name}_compile_scenario_minemission.csv')
    df.to_csv(csv_file_path, index=False)
    
    return df


def plot_scenario_minemission(df, case_study, obj_mode, scenario_name):
    import os
    fig, ax = plt.subplots(1, 5, figsize=(20, 5))
    # Plot and label the first subplot for harvested area
    ax[0].bar(df.period, df.oha)
    ax[0].set_ylim(0, None)
    ax[0].set_title('Harvested area')
    ax[0].set_xlabel('Period')
    ax[0].set_ylabel('Area (ha)')
    
    # Plot and label the second subplot for harvested volume
    ax[1].bar(df.period, df.ohv)
    ax[1].set_ylim(0, None)
    ax[1].set_title('Harvested volume')
    ax[1].set_xlabel('Period')
    ax[1].set_ylabel('Volume (m3)')

    # Plot and label the third subplot for growing stock
    ax[2].bar(df.period, df.ogs)
    ax[2].set_ylim(0, None)
    ax[2].set_xlabel('Period')
    ax[2].set_title('Growing Stock')
    ax[2].set_ylabel('Volume (m3)')

    # Plot and label the fourth subplot for ecosystem carbon stock
    ax[3].bar(df.period, df.ocp)
    ax[3].set_ylim(0, None)
    ax[3].set_title('Ecosystem C stock')
    ax[3].set_xlabel('Period')
    ax[3].set_ylabel('Stock (ton)')

    # # Plot and label the fifth subplot for total carbon emission
    ax[4].bar(df.period, df.ocf)
    ax[4].set_ylim(None, None)
    ax[4].set_title('Net Carbon Emission')
    ax[4].set_xlabel('Period')
    ax[4].set_ylabel('tons of CO2')

    plt.tight_layout()
    
    folder_path = os.path.join('./outputs/fig', case_study)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)   
    file_name = f"{case_study}_{obj_mode}_{scenario_name}_scheduling_minemission.pdf"
    file_path = os.path.join(folder_path, file_name)
    
    # Save and show plot
    plt.savefig(file_path)
    plt.show()
    plt.close()
    print(f"Plot saved to {file_path}")
    
    return fig, ax



################################################
# Optimization
################################################

def cmp_c_ss(fm, path, clt_percentage, hwp_pool_effect_value, expr, yname, half_life_solid_wood=30, half_life_paper=2, half_life_clt= float('inf'), proportion_solid_wood=0.8, util=0.85, mask=None):
    """
    Compile objective function coefficient for total system carbon stock indicators (given ForestModel instance, 
    leaf-to-root-node path, and expression to evaluate).
    """
    k_wood = math.log(2) / half_life_solid_wood  # Decay rate for solid wood products (30-year half-life)
    k_paper = math.log(2) / half_life_paper  # Decay rate for paper (2-year half-life)
    k_clt = math.log(2) / half_life_clt  # Decay rate for clt (INF half-life)
    wood_density = 460 #kg/m3
    carbon_content = 0.5
    result = 0.
    sum = 0.
    hwp_accu_wood = 0.
    hwp_accu_paper = 0.
    hwp_accu_clt = 0.
    ecosystem = 0.
    for t, n in enumerate(path, start=1):        
        d = n.data()    
        if fm.is_harvest(d['acode']):
            result_hwp = fm.compile_product(t, 'totvol', d['acode'], [d['dtk']], d['age'], coeff=False) * wood_density * carbon_content/1000
        else:
            result_hwp = 0     
        hwp_accu_wood  = hwp_accu_wood * (1-k_wood)**10 + result_hwp * util * proportion_solid_wood * (1-clt_percentage)
        hwp_accu_paper = hwp_accu_paper * (1-k_paper)**10 + result_hwp * util * (1- proportion_solid_wood) 
        hwp_accu_clt = hwp_accu_clt * (1-k_clt)**10 + result_hwp * util * clt_percentage * proportion_solid_wood 

        
        ecosystem = fm.inventory(t, yname, age=d['_age'], dtype_keys=[d['_dtk']])
        result = hwp_pool_effect_value * (hwp_accu_wood + hwp_accu_paper + hwp_accu_clt) + ecosystem

    return result




def cmp_c_se(fm, path, clt_percentage, hwp_pool_effect_value, displacement_effect, release_immediately_value, expr, yname, half_life_solid_wood=30, half_life_paper=2, proportion_solid_wood=0.8, util=0.85, mask=None):
    """
    Compile objective function coefficient for total system emission indicators (given ForestModel instance, 
    leaf-to-root-node path, and expression to evaluate).
    """
    k_wood = math.log(2) / half_life_solid_wood  # Decay rate for solid wood products (30-year half-life)
    k_paper = math.log(2) / half_life_paper  # Decay rate for paper (2-year half-life)
    # k_wood = 0.
    # k_paper = 0.
    wood_density = 460 #kg/m3
    carbon_content = 0.5 # percent
    result = 0.
    hwp_wood_emission = 0.
    hwp_paper_emission = 0.
    hwp_emission_immediately = 0.
    hwps_residue_pool = 0.
    hwp_accu_wood = 0.
    hwp_accu_wood = 0.
    hwp_accu_paper = 0.
    ecosystem = 0.
    credibility = 1.
    clt_conversion_rate = 1.24
    co2_concrete_manu_accu = 0.
    co2_concrete_landfill = 0.
    co2_concrete_manu_factor = 298. #kg/m3
    concrete_density = 2400 #kg/m3
    co2_concrete_landfill_factor = 0.00517 * concrete_density
    co2_concrete_landfill_accu = 0.
    
    for t, n in enumerate(path, start=1):        
        d = n.data()     
        if fm.is_harvest(d['acode']):
            result_hwp = fm.compile_product(t, 'totvol', d['acode'], [d['dtk']], d['age'], coeff=False) * wood_density * carbon_content  #kg/m3
            concrete_volume = fm.compile_product(t, 'totvol', d['acode'], [d['dtk']], d['age'], coeff=False) * proportion_solid_wood * clt_percentage * credibility / clt_conversion_rate 

        else:
            result_hwp = 0.  
            concrete_volume = 0.
        hwp_accu_wood  = hwp_accu_wood * (1-k_wood)**10 + result_hwp  * util * proportion_solid_wood * (1 - clt_percentage)
        hwp_accu_paper = hwp_accu_paper * (1-k_paper)**10 + result_hwp * util * (1- proportion_solid_wood) 
        hwps_residue_pool = result_hwp * (1.0 - util)


        
        hwp_wood_emission  =  (hwp_accu_wood * (1- (1-k_wood)**10)  * 44/12) /1000.
        hwp_paper_emission =  (hwp_accu_paper * (1- (1-k_paper)**10) * 44/12 ) /1000.
        hwps_residue_emission = (hwps_residue_pool * 44/12) /1000.

        hwp_emission_immediately =  (result_hwp  * util * 44/12 )/1000


        
        net_emission = 10 * fm.inventory(t, yname, age=d['_age'], dtype_keys=[d['_dtk']])
        
        co2_concrete_manu_accu += concrete_volume * util * co2_concrete_manu_factor / 1000.
        co2_concrete_landfill_accu += concrete_volume * util * co2_concrete_landfill_factor / 1000.
        
        result += hwp_pool_effect_value * (hwp_wood_emission + hwp_paper_emission) + net_emission + hwps_residue_emission + release_immediately_value * hwp_emission_immediately - displacement_effect * (co2_concrete_manu_accu + co2_concrete_landfill_accu)

    return result


def cmp_c_z(fm, path, expr):
    """
    Compile objective function coefficient (given ForestModel instance, 
    leaf-to-root-node path, and expression to evaluate).
    """
    result = 0.
    for t, n in enumerate(path, start=1):        
        d = n.data()    
        if fm.is_harvest(d['acode']):
            result += fm.compile_product(t, expr, d['acode'], [d['dtk']], d['age'], coeff=False)
    return result

def cmp_c_cflw(fm, path, expr, mask=None): # product, all harvest actions
    """
    Compile flow constraint coefficient for product indicator (given ForestModel 
    instance, leaf-to-root-node path, expression to evaluate, and optional mask).
    """
    result = {}
    for t, n in enumerate(path, start=1):
        d = n.data()
        if mask and not fm.match_mask(mask, d['dtk']): continue
        if fm.is_harvest(d['acode']):
            result[t] = fm.compile_product(t, expr, d['acode'], [d['dtk']], d['age'], coeff=False)
    return result


def cmp_c_caa(fm, path, expr, acodes, mask=None): # product, named actions
    """
    Compile constraint coefficient for product indicator (given ForestModel 
    instance, leaf-to-root-node path, expression to evaluate, list of action codes, 
    and optional mask).
    """
    result = {}
    for t, n in enumerate(path, start=1):
        d = n.data()
        if mask and not fm.match_mask(mask, d['dtk']): continue
        if d['acode'] in acodes:
            result[t] = fm.compile_product(t, expr, d['acode'], [d['dtk']], d['age'], coeff=False)
    return result


def cmp_c_ci(fm, path, yname, mask=None): # product, named actions
    """
    Compile constraint coefficient for inventory indicator (given ForestModel instance, 
    leaf-to-root-node path, expression to evaluate, and optional mask).
    """
    result = {}
    for t, n in enumerate(path, start=1):
        d = n.data()
        if mask and not fm.match_mask(mask, d['_dtk']): continue
        result[t] = fm.inventory(t, yname=yname, age=d['_age'], dtype_keys=[d['_dtk']]) 
        #result[t] = fm.inventory(t, yname=yname, age=d['age'], dtype_keys=[d['dtk']]) 
    return result


def gen_scenario(fm, clt_percentage=1.0,hwp_pool_effect_value=1.0, displacement_effect=1.0, release_immediately_value=0., name='base', util=0.85, harvest_acode='harvest',
                 cflw_ha={}, cflw_hv={}, 
                 cgen_ha={}, cgen_hv={}, cgen_gs={}, 
                 tvy_name='totvol', cp_name='ecosystem', ce_name='net_emission', obj_mode='max_hv', mask=None):
    from functools import partial
    import numpy as np
    coeff_funcs = {}
    cflw_e = {}
    cgen_data = {}
    acodes = ['null', harvest_acode] # define list of action codes
    vexpr = '%s * %0.2f' % (tvy_name, util) # define volume expression
    if obj_mode == 'max_hv': # maximize harvest volume
        sense = ws3.opt.SENSE_MAXIMIZE 
        zexpr = vexpr
    elif obj_mode == 'max_st':
        sense = ws3.opt.SENSE_MAXIMIZE 
        zexpr = vexpr 
    elif obj_mode == 'min_em':
        sense = ws3.opt.SENSE_MINIMIZE 
        zexpr = vexpr 
    elif obj_mode == 'min_ha': # minimize harvest area
        sense = ws3.opt.SENSE_MINIMIZE 
        zexpr = '1.'
    else:
        raise ValueError('Invalid obj_mode: %s' % obj_mode)        
    if obj_mode == 'max_hv':
        coeff_funcs['z'] = partial(cmp_c_z, expr=zexpr) # define objective function coefficient function for max_hv and min_ha
    elif obj_mode == 'min_ha':
        coeff_funcs['z'] = partial(cmp_c_z, expr=zexpr) # define objective function coefficient function for max_hv and min_ha
    elif obj_mode == 'max_st':
        coeff_funcs['z'] = partial(cmp_c_ss, clt_percentage=clt_percentage, hwp_pool_effect_value=hwp_pool_effect_value, expr=zexpr, yname=cp_name) # define objective function coefficient function for total system carbon stock
    elif obj_mode == 'min_em':
        coeff_funcs['z'] = partial(cmp_c_se, clt_percentage=clt_percentage, hwp_pool_effect_value=hwp_pool_effect_value, displacement_effect=displacement_effect, release_immediately_value=release_immediately_value,expr=zexpr, yname=ce_name) # define objective function coefficient function for total system emission
    else:
        raise ValueError('Invalid obj_mode: %s' % obj_mode)

    T = fm.periods
    if cflw_ha: # define even flow constraint (on harvest area)
        cname = 'cflw_ha'
        coeff_funcs[cname] = partial(cmp_c_caa, expr='1.', acodes=[harvest_acode], mask=None) 
        cflw_e[cname] = cflw_ha
    if cflw_hv: # define even flow constraint (on harvest volume)
        cname = 'cflw_hv'
        coeff_funcs[cname] = partial(cmp_c_caa, expr=vexpr, acodes=[harvest_acode], mask=None) 
        cflw_e[cname] = cflw_hv         
    if cgen_ha: # define general constraint (harvest area)
        cname = 'cgen_ha'
        coeff_funcs[cname] = partial(cmp_c_caa, expr='1.', acodes=[harvest_acode], mask=None) 
        cgen_data[cname] = cgen_ha
    if cgen_hv: # define general constraint (harvest volume)
        cname = 'cgen_hv'
        coeff_funcs[cname] = partial(cmp_c_caa, expr=vexpr, acodes=[harvest_acode], mask=None) 
        cgen_data[cname] = cgen_hv
    if cgen_gs: # define general constraint (growing stock)
        cname = 'cgen_gs'
        coeff_funcs[cname] = partial(cmp_c_ci, yname=tvy_name, mask=None)
        cgen_data[cname] = cgen_gs
    return fm.add_problem(name, coeff_funcs, cflw_e, cgen_data=cgen_data, acodes=acodes, sense=sense, mask=mask)



def run_scenario(fm, clt_percentage, hwp_pool_effect_value, displacement_effect, release_immediately_value, case_study, obj_mode, scenario_name='no_cons', solver=ws3.opt.SOLVER_PULP):
    import gurobipy as grb
    # initial_inv_ms1 = 869737. #ha
    initial_gs_ms1 = 106582957.  #m3   
    # initial_inv_ms3 = 390738. #ha
    initial_gs_ms3 =18018809. #m3   
    # initial_inv_ms2 = 191273. #ha
    initial_gs_ms2 = 7017249. #m3   
    aac_ms1 =  18255528. # AAC per year * 10
    aac_ms3 =  1072860. # AAC per year * 10
    aac_ms2 =  766066. # AAC per year * 10
    cflw_ha = {}
    cflw_hv = {}
    cgen_ha = {}
    cgen_hv = {}
    cgen_gs = {}

    if scenario_name == 'no_cons': 
        # no constraints scenario : 
        print('running no_cons scenario')
    
    elif scenario_name == 'evenflow_cons': 
        # no_cons scenario : 
        print('running even flow constraints scenario')
        cflw_ha = ({p:0.05 for p in fm.periods}, 1)
        cflw_hv = ({p:0.05 for p in fm.periods}, 1)
 
    # Red mining_site_3
    elif scenario_name == 'bau_ms3': 
        # Business as usual scenario for the mining_site_3: 
        print('running business as usual scenario for the mining site, 3')
        cflw_ha = ({p:0.05 for p in fm.periods}, 1)
        cflw_hv = ({p:0.05 for p in fm.periods}, 1)
        cgen_hv = {'lb':{1:aac_ms3}, 'ub':{1:aac_ms3}} # Equal with Annual Allowable Cut
        cgen_gs = {'lb':{10:initial_gs_ms3*0.9}, 'ub':{10:initial_gs_ms3*1000}} #Not less than 90% of initial growing stock
    elif scenario_name == 'ms3_AAC_90%': 
        # Alternative scenario for the mining_site_3 (90%_AAC): 
        print('running the scenario for the mining site 3(90%_AAC),')
        cflw_ha = ({p:0.05 for p in fm.periods}, 1)
        cflw_hv = ({p:0.05 for p in fm.periods}, 1)
        cgen_hv = {'lb':{1:0.9*aac_ms3}, 'ub':{1:0.9*aac_ms3+1}} # Equal with Annual Allowable Cut
        cgen_gs = {'lb':{10:initial_gs_ms3*0.9}, 'ub':{10:initial_gs_ms3*1000}} #Not less than 90% of initial growing stock
    elif scenario_name == 'ms3_AAC_80%': 
        # Alternative scenario for the mining_site_3 (80%_AAC): 
        print('running the scenario for the mining site 3 (80%_AAC),')
        cflw_ha = ({p:0.05 for p in fm.periods}, 1)
        cflw_hv = ({p:0.05 for p in fm.periods}, 1)
        cgen_hv = {'lb':{1:0.8*aac_ms3}, 'ub':{1:0.8*aac_ms3+1}} # Equal with Annual Allowable Cut
        cgen_gs = {'lb':{10:initial_gs_ms3*0.9}, 'ub':{10:initial_gs_ms3*1000}} #Not less than 90% of initial growing stock
    elif scenario_name == 'ms3_AAC_70%': 
        # Alternative scenario for the mining_site_3 (70%_AAC): 
        print('running the scenario for the mining site 3 (70%_AAC),')
        cflw_ha = ({p:0.05 for p in fm.periods}, 1)
        cflw_hv = ({p:0.05 for p in fm.periods}, 1)
        cgen_hv = {'lb':{1:0.7*aac_ms3}, 'ub':{1:0.7*aac_ms3+1}} # Equal with Annual Allowable Cut
        cgen_gs = {'lb':{10:initial_gs_ms3*0.9}, 'ub':{10:initial_gs_ms3*1000}} #Not less than 90% of initial growing stock
    elif scenario_name == 'ms3_AAC_60%': 
        # Alternative scenario for the mining_site_3 (60%_AAC): 
        print('running the scenario for the mining site 3 (60%_AAC),')
        cflw_ha = ({p:0.05 for p in fm.periods}, 1)
        cflw_hv = ({p:0.05 for p in fm.periods}, 1)
        cgen_hv = {'lb':{1:0.6*aac_ms3}, 'ub':{1:0.6*aac_ms3+1}} # Equal with Annual Allowable Cut
        cgen_gs = {'lb':{10:initial_gs_ms3*0.9}, 'ub':{10:initial_gs_ms3*1000}} #Not less than 90% of initial growing stock
    elif scenario_name == 'ms3_AAC_50%': 
        # Alternative scenario for the mining_site_3 (50%_AAC): 
        print('running the scenario for the mining site 3 (50%_AAC),')
        cflw_ha = ({p:0.05 for p in fm.periods}, 1)
        cflw_hv = ({p:0.05 for p in fm.periods}, 1)
        cgen_hv = {'lb':{1:0.5*aac_ms3}, 'ub':{1:0.5*aac_ms3+1}} # Equal with Annual Allowable Cut
        cgen_gs = {'lb':{10:initial_gs_ms3*0.9}, 'ub':{10:initial_gs_ms3*1000}} #Not less than 90% of initial growing stock
    elif scenario_name == 'ms3_AAC_40%': 
        # Alternative scenario for the mining_site_3 (40%_AAC): 
        print('running the scenario for the mining site 3 (40%_AAC),')
        cflw_ha = ({p:0.05 for p in fm.periods}, 1)
        cflw_hv = ({p:0.05 for p in fm.periods}, 1)
        cgen_hv = {'lb':{1:0.4*aac_ms3}, 'ub':{1:0.4*aac_ms3+1}} # Equal with Annual Allowable Cut
        cgen_gs = {'lb':{10:initial_gs_ms3*0.9}, 'ub':{10:initial_gs_ms3*1000}} #Not less than 90% of initial growing stock
    elif scenario_name == 'ms3_AAC_30%': 
        # Alternative scenario for the mining_site_3 (30%_AAC): 
        print('running the scenario for the  mining site 3 (30%_AAC),')
        cflw_ha = ({p:0.05 for p in fm.periods}, 1)
        cflw_hv = ({p:0.05 for p in fm.periods}, 1)
        cgen_hv = {'lb':{1:0.3*aac_ms3}, 'ub':{1:0.3*aac_ms3+1}} # Equal with Annual Allowable Cut
        cgen_gs = {'lb':{10:initial_gs_ms3*0.9}, 'ub':{10:initial_gs_ms3*1000}} #Not less than 90% of initial growing stock
    elif scenario_name == 'ms3_AAC_20%': 
        # Alternative scenario for the mining_site_3 (20%_AAC): 
        print('running the scenario for the  mining site 3 (20%_AAC),')
        cflw_ha = ({p:0.05 for p in fm.periods}, 1)
        cflw_hv = ({p:0.05 for p in fm.periods}, 1)
        cgen_hv = {'lb':{1:0.2*aac_ms3}, 'ub':{1:0.2*aac_ms3+1}} # Equal with Annual Allowable Cut
        cgen_gs = {'lb':{10:initial_gs_ms3*0.9}, 'ub':{10:initial_gs_ms3*1000}} #Not less than 90% of initial growing stock
    elif scenario_name == 'ms3_AAC_10%': 
        # Alternative scenario for the mining_site_3 (10%_AAC): 
        print('running the scenario for the mining site 3 (10%_AAC),')
        cflw_ha = ({p:0.05 for p in fm.periods}, 1)
        cflw_hv = ({p:0.05 for p in fm.periods}, 1)
        cgen_hv = {'lb':{1:0.1*aac_ms3}, 'ub':{1:0.1*aac_ms3+1}} # Equal with Annual Allowable Cut
        cgen_gs = {'lb':{10:initial_gs_ms3*0.9}, 'ub':{10:initial_gs_ms3*1000}} #Not less than 90% of initial growing stock
  
    # the mining site 2 scenarios
    elif scenario_name == 'bau_ms2': 
        # Business as usual scenario for the mining site 2: 
        print('running business as usual scenario for the mine site 2')
        cgen_hv = {'lb':{1:aac_ms2}, 'ub':{1:aac_ms2}}
        cflw_ha = ({p:0.05 for p in fm.periods}, 1)
        cflw_hv = ({p:0.05 for p in fm.periods}, 1)
        cgen_gs = {'lb':{10:initial_gs_ms2*0.9}, 'ub':{10:initial_gs_ms2*1000}} #Not less than 90% of initial growing stock
    elif scenario_name == 'ms2_AAC_90%': 
        # Alternative scenario for the mining site 2 (90%_AAC): 
        print('running the scenario for the mining site 2 (90%_AAC),')
        cflw_ha = ({p:0.05 for p in fm.periods}, 1)
        cflw_hv = ({p:0.05 for p in fm.periods}, 1)
        cgen_hv = {'lb':{1:0.9*aac_ms2}, 'ub':{1:0.9*aac_ms2+1}} # Equal with Annual Allowable Cut
        cgen_gs = {'lb':{10:initial_gs_ms2*0.9}, 'ub':{10:initial_gs_ms2*1000}} #Not less than 90% of initial growing stock
    elif scenario_name == 'ms2_AAC_80%': 
        # Alternative scenario for the mining site 2 (80%_AAC): 
        print('running the scenario for the mining site 2 (80%_AAC),')
        cflw_ha = ({p:0.05 for p in fm.periods}, 1)
        cflw_hv = ({p:0.05 for p in fm.periods}, 1)
        cgen_hv = {'lb':{1:0.8*aac_ms2}, 'ub':{1:0.8*aac_ms2+1}} # Equal with Annual Allowable Cut
        cgen_gs = {'lb':{10:initial_gs_ms2*0.9}, 'ub':{10:initial_gs_ms2*1000}} #Not less than 90% of initial growing stock    
    elif scenario_name == 'ms2_AAC_70%': 
        # Alternative scenario for the mining site 2 (70%_AAC): 
        print('running the scenario for the mining site 2 (70%_AAC),')
        cflw_ha = ({p:0.05 for p in fm.periods}, 1)
        cflw_hv = ({p:0.05 for p in fm.periods}, 1)
        cgen_hv = {'lb':{1:0.7*aac_ms2}, 'ub':{1:0.7*aac_ms2+1}} # Equal with Annual Allowable Cut
        cgen_gs = {'lb':{10:initial_gs_ms2*0.9}, 'ub':{10:initial_gs_ms2*1000}} #Not less than 90% of initial growing stock     
    elif scenario_name == 'ms2_AAC_60%': 
        # Alternative scenario for the mining site 2 (60%_AAC): 
        print('running the scenario for the mining site 2 (60%_AAC),')
        cflw_ha = ({p:0.05 for p in fm.periods}, 1)
        cflw_hv = ({p:0.05 for p in fm.periods}, 1)
        cgen_hv = {'lb':{1:0.6*aac_ms2}, 'ub':{1:0.6*aac_ms2+1}} # Equal with Annual Allowable Cut
        cgen_gs = {'lb':{10:initial_gs_ms2*0.9}, 'ub':{10:initial_gs_ms2*1000}} #Not less than 90% of initial growing stock 
    elif scenario_name == 'ms2_AAC_50%': 
        # Alternative scenario for the mining site 2 (50%_AAC): 
        print('running the scenario for the mining site 2 (50%_AAC),')
        cflw_ha = ({p:0.05 for p in fm.periods}, 1)
        cflw_hv = ({p:0.05 for p in fm.periods}, 1)
        cgen_hv = {'lb':{1:0.5*aac_ms2}, 'ub':{1:0.5*aac_ms2+1}} # Equal with Annual Allowable Cut
        cgen_gs = {'lb':{10:initial_gs_ms2*0.9}, 'ub':{10:initial_gs_ms2*1000}} #Not less than 90% of initial growing stock     
    elif scenario_name == 'ms2_AAC_40%': 
        # Alternative scenario for the  mining site 2 (40%_AAC): 
        print('running the scenario for the  mining site 2 (40%_AAC),')
        cflw_ha = ({p:0.05 for p in fm.periods}, 1)
        cflw_hv = ({p:0.05 for p in fm.periods}, 1)
        cgen_hv = {'lb':{1:0.4*aac_ms2}, 'ub':{1:0.4*aac_ms2+1}} # Equal with Annual Allowable Cut
        cgen_gs = {'lb':{10:initial_gs_ms2*0.9}, 'ub':{10:initial_gs_ms2*1000}} #Not less than 90% of initial growing stock     
    elif scenario_name == 'ms2_AAC_30%': 
        # Alternative scenario for the mining site 2 (30%_AAC): 
        print('running the scenario for the mining site 2 (30%_AAC),')
        cflw_ha = ({p:0.05 for p in fm.periods}, 1)
        cflw_hv = ({p:0.05 for p in fm.periods}, 1)
        cgen_hv = {'lb':{1:0.3*aac_ms2}, 'ub':{1:0.3*aac_ms2+1}} # Equal with Annual Allowable Cut
        cgen_gs = {'lb':{10:initial_gs_ms2*0.9}, 'ub':{10:initial_gs_ms2*1000}} #Not less than 90% of initial growing stock     
    elif scenario_name == 'ms2_AAC_20%': 
        # Alternative scenario for the  mining site 2 (20%_AAC): 
        print('running the scenario for the  mining site 2 (20%_AAC),')
        cflw_ha = ({p:0.05 for p in fm.periods}, 1)
        cflw_hv = ({p:0.05 for p in fm.periods}, 1)
        cgen_hv = {'lb':{1:0.2*aac_ms2}, 'ub':{1:0.2*aac_ms2+1}} # Equal with Annual Allowable Cut
        cgen_gs = {'lb':{10:initial_gs_ms2*0.9}, 'ub':{10:initial_gs_ms2*1000}} #Not less than 90% of initial growing stock     
    elif scenario_name == 'ms2_AAC_10%': 
        # Alternative scenario for the  mining site 2 (10%_AAC): 
        print('running the scenario for the mining site 2 (10%_AAC),')
        cflw_ha = ({p:0.05 for p in fm.periods}, 1)
        cflw_hv = ({p:0.05 for p in fm.periods}, 1)
        cgen_hv = {'lb':{1:0.1*aac_ms2}, 'ub':{1:0.1*aac_ms2+1}} # Equal with Annual Allowable Cut
        cgen_gs = {'lb':{10:initial_gs_ms2*0.9}, 'ub':{10:initial_gs_ms2*1000}} #Not less than 90% of initial growing stock 
    
    # the mining site 1 scenarios
    elif scenario_name == 'bau_ms1': 
        # Business as usual scenario for the mining site 1 : 
        print('running business as usual scenario for the mining site 1 ')
        cflw_ha = ({p:0.05 for p in fm.periods}, 1)
        cflw_hv = ({p:0.05 for p in fm.periods}, 1)
        cgen_hv = {'lb':{1:0.50*aac_ms1}, 'ub':{1:0.50*aac_ms1+1}} 
        cgen_gs = {'lb':{10:initial_gs_ms1*0.9}, 'ub':{10:initial_gs_ms1*1000}} #Not less than 90% of initial growing stock
    elif scenario_name == 'ms1_AAC_90%': 
        # Alternative scenario for the  mining site 1 (90%_AAC): 
        print('running the scenario for the  mining site 1 (90%_AAC),')
        cflw_ha = ({p:0.05 for p in fm.periods}, 1)
        cflw_hv = ({p:0.05 for p in fm.periods}, 1)
        cgen_hv = {'lb':{1:0.9*0.5*aac_ms1}, 'ub':{1:0.9*0.5*aac_ms1+1}} 
        cgen_gs = {'lb':{10:initial_gs_ms1*0.9}, 'ub':{10:initial_gs_ms1*1000}} #Not less than 90% of initial growing stock
    elif scenario_name == 'ms1_AAC_80%': 
        # Alternative scenario for the  mining site 1 (80%_AAC): 
        print('running the scenario for the  mining site 1 (80%_AAC),')
        cflw_ha = ({p:0.05 for p in fm.periods}, 1)
        cflw_hv = ({p:0.05 for p in fm.periods}, 1)
        cgen_hv = {'lb':{1:0.8*0.5*aac_ms1}, 'ub':{1:0.8*0.5*aac_ms1+1}} 
        cgen_gs = {'lb':{10:initial_gs_ms1*0.9}, 'ub':{10:initial_gs_ms1*1000}} #Not less than 90% of initial growing stock    
    elif scenario_name == 'ms1_AAC_70%': 
        # Alternative scenario for the  mining site 1 (70%_AAC): 
        print('running the scenario for the mining site 1 (70%_AAC),')
        cflw_ha = ({p:0.05 for p in fm.periods}, 1)
        cflw_hv = ({p:0.05 for p in fm.periods}, 1)
        cgen_hv = {'lb':{1:0.7*0.5*aac_ms1}, 'ub':{1:0.7*0.5*aac_ms1+1}} 
        cgen_gs = {'lb':{10:initial_gs_ms1*0.9}, 'ub':{10:initial_gs_ms1*1000}} #Not less than 90% of initial growing stock   
    elif scenario_name == 'ms1_AAC_60%': 
        # Alternative scenario for the  mining site 1 (60%_AAC): 
        print('running the scenario for the  mining site 1 (60%_AAC),')
        cflw_ha = ({p:0.05 for p in fm.periods}, 1)
        cflw_hv = ({p:0.05 for p in fm.periods}, 1)
        cgen_hv = {'lb':{1:0.6*0.5*aac_ms1}, 'ub':{1:0.6*0.5*aac_ms1+1}} 
        cgen_gs = {'lb':{10:initial_gs_ms1*0.9}, 'ub':{10:initial_gs_ms1*1000}} #Not less than 90% of initial growing stock           
    elif scenario_name == 'ms1_AAC_50%': 
        # Alternative scenario for the  mining site 1 (50%_AAC): 
        print('running the scenario for the mining site 1 (50%_AAC),')
        cflw_ha = ({p:0.05 for p in fm.periods}, 1)
        cflw_hv = ({p:0.05 for p in fm.periods}, 1)
        cgen_hv = {'lb':{1:0.5*0.5*aac_ms1}, 'ub':{1:0.5*0.5*aac_ms1+1}} 
        cgen_gs = {'lb':{10:initial_gs_ms1*0.9}, 'ub':{10:initial_gs_ms1*1000}} #Not less than 90% of initial growing stock   
    elif scenario_name == 'ms1_AAC_40%': 
        # Alternative scenario for the  mining site 1 (40%_AAC): 
        print('running the scenario for the  mining site 1 (40%_AAC),')
        cflw_ha = ({p:0.05 for p in fm.periods}, 1)
        cflw_hv = ({p:0.05 for p in fm.periods}, 1)
        cgen_hv = {'lb':{1:0.4*0.5*aac_ms1}, 'ub':{1:0.4*0.5*aac_ms1+1}} 
        cgen_gs = {'lb':{10:initial_gs_ms1*0.9}, 'ub':{10:initial_gs_ms1*1000}} #Not less than 90% of initial growing stock   
    elif scenario_name == 'ms1_AAC_30%': 
        # Alternative scenario for the mining site 1 (30%_AAC): 
        print('running the scenario for the  mining site 1 (30%_AAC),')
        cflw_ha = ({p:0.05 for p in fm.periods}, 1)
        cflw_hv = ({p:0.05 for p in fm.periods}, 1)
        cgen_hv = {'lb':{1:0.3*0.5*aac_ms1}, 'ub':{1:0.3*0.5*aac_ms1+1}} 
        cgen_gs = {'lb':{10:initial_gs_ms1*0.9}, 'ub':{10:initial_gs_ms1*1000}} #Not less than 90% of initial growing stock   
    elif scenario_name == 'ms1_AAC_20%': 
        # Alternative scenario for the mining site 1 (20%_AAC): 
        print('running the scenario for the mining site 1 (20%_AAC),')
        cflw_ha = ({p:0.05 for p in fm.periods}, 1)
        cflw_hv = ({p:0.05 for p in fm.periods}, 1)
        cgen_hv = {'lb':{1:0.2*0.5*aac_ms1}, 'ub':{1:0.2*0.5*aac_ms1+1}} 
        cgen_gs = {'lb':{10:initial_gs_ms1*0.9}, 'ub':{10:initial_gs_ms1*1000}} #Not less than 90% of initial growing stock       
    elif scenario_name == 'ms1_AAC_10%': 
        # Alternative scenario for the  mining site 1 (10%_AAC): 
        print('running the scenario for the  mining site 1 (10%_AAC),')
        cflw_ha = ({p:0.05 for p in fm.periods}, 1)
        cflw_hv = ({p:0.05 for p in fm.periods}, 1)
        cgen_hv = {'lb':{1:0.1*0.5*aac_ms1}, 'ub':{1:0.1*0.5*aac_ms1+1}} 
        cgen_gs = {'lb':{10:initial_gs_ms1*0.9}, 'ub':{10:initial_gs_ms1*1000}} #Not less than 90% of initial growing stock  
    else:
        assert False # bad scenario name
    
    p = gen_scenario(fm=fm, 
                     clt_percentage=clt_percentage,
                     hwp_pool_effect_value=hwp_pool_effect_value,
                     displacement_effect=displacement_effect,
                     release_immediately_value=release_immediately_value,
                     name=scenario_name, 
                     cflw_ha=cflw_ha, 
                     cflw_hv=cflw_hv,
                     cgen_ha=cgen_ha,
                     cgen_hv=cgen_hv,
                     cgen_gs=cgen_gs,
                    obj_mode=obj_mode)
    p.solver(solver)
    
    fm.reset()
    p.solve()
    
    if p.status() != ws3.opt.STATUS_OPTIMAL:
        print('Model not optimal.')
        df = None   
    else:
        sch = fm.compile_schedule(p)
        fm.apply_schedule(sch, 
                        force_integral_area=False, 
                        override_operability=False,
                        fuzzy_age=False,
                        recourse_enabled=False,
                        verbose=False,
                        compile_c_ycomps=True)
    
    if obj_mode == 'max_hv' or obj_mode == 'min_ha':
        df = compile_scenario(fm, case_study, obj_mode, scenario_name)
        plot_scenario(df, case_study, obj_mode, scenario_name)
    elif obj_mode == 'max_st':
        df = compile_scenario_maxstock(fm, case_study, obj_mode, scenario_name)
        plot_scenario_maxstock(df, case_study, obj_mode, scenario_name) 
    elif obj_mode == 'min_em':
        df = compile_scenario_minemission(fm, case_study, obj_mode, scenario_name)
        plot_scenario_minemission(df, case_study, obj_mode, scenario_name)
    else:
        raise ValueError('Invalid obj_mode: %s' % obj_mode) 
        
    print("------------------------------------------------")
    kpi_socioeconomic(fm)
    print("------------------------------------------------")


    return sch


##############################################################
# Implement a simple function to run CBM from ws3 export data
##############################################################
def run_cbm_emissionstock(sit_config, sit_tables, n_steps):
    from libcbm.input.sit import sit_reader
    from libcbm.input.sit import sit_cbm_factory 
    from libcbm.model.cbm.cbm_output import CBMOutput
    from libcbm.storage.backends import BackendType
    from libcbm.model.cbm import cbm_simulator
    sit_data = sit_reader.parse(sit_classifiers=sit_tables['sit_classifiers'],
                                sit_disturbance_types=sit_tables['sit_disturbance_types'],
                                sit_age_classes=sit_tables['sit_age_classes'],
                                sit_inventory=sit_tables['sit_inventory'],
                                sit_yield=sit_tables['sit_yield'],
                                sit_events=sit_tables['sit_events'],
                                sit_transitions=sit_tables['sit_transitions'],
                                sit_eligibilities=None)
    sit = sit_cbm_factory.initialize_sit(sit_data=sit_data, config=sit_config)
    classifiers, inventory = sit_cbm_factory.initialize_inventory(sit)
    cbm_output = CBMOutput(classifier_map=sit.classifier_value_names,
                           disturbance_type_map=sit.disturbance_name_map)
    with sit_cbm_factory.initialize_cbm(sit) as cbm:
        # Create a function to apply rule based disturbance events and transition rules based on the SIT input
        rule_based_processor = sit_cbm_factory.create_sit_rule_based_processor(sit, cbm)
        # The following line of code spins up the CBM inventory and runs it through 200 timesteps.
        cbm_simulator.simulate(cbm,
                               n_steps=n_steps,
                               classifiers=classifiers,
                               inventory=inventory,
                               pre_dynamics_func=rule_based_processor.pre_dynamics_func,
                               reporting_func=cbm_output.append_simulation_result,
                               backend_type=BackendType.numpy)      
    return cbm_output

def run_cbm(df_carbon_stock, df_carbon_emission, df_carbon_emission_immed, df_emission_concrete_manu, df_emission_concrete_landfill, sit_config, sit_tables, n_steps, release_immediately_value, plot=True):
    from libcbm.input.sit import sit_reader
    from libcbm.input.sit import sit_cbm_factory 
    from libcbm.model.cbm.cbm_output import CBMOutput
    from libcbm.storage.backends import BackendType
    from libcbm.model.cbm import cbm_simulator
    sit_data = sit_reader.parse(sit_classifiers=sit_tables['sit_classifiers'],
                                sit_disturbance_types=sit_tables['sit_disturbance_types'],
                                sit_age_classes=sit_tables['sit_age_classes'],
                                sit_inventory=sit_tables['sit_inventory'],
                                sit_yield=sit_tables['sit_yield'],
                                sit_events=sit_tables['sit_events'],
                                sit_transitions=sit_tables['sit_transitions'],
                                sit_eligibilities=None)
    sit = sit_cbm_factory.initialize_sit(sit_data=sit_data, config=sit_config)
    classifiers, inventory = sit_cbm_factory.initialize_inventory(sit)
    cbm_output = CBMOutput(classifier_map=sit.classifier_value_names,
                           disturbance_type_map=sit.disturbance_name_map)
    with sit_cbm_factory.initialize_cbm(sit) as cbm:
        # Create a function to apply rule based disturbance events and transition rules based on the SIT input
        rule_based_processor = sit_cbm_factory.create_sit_rule_based_processor(sit, cbm)
        # The following line of code spins up the CBM inventory and runs it through 200 timesteps.
        cbm_simulator.simulate(cbm,
                               n_steps=n_steps,
                               classifiers=classifiers,
                               inventory=inventory,
                               pre_dynamics_func=rule_based_processor.pre_dynamics_func,
                               reporting_func=cbm_output.append_simulation_result,
                               backend_type=BackendType.numpy)
    pi = cbm_output.classifiers.to_pandas().merge(cbm_output.pools.to_pandas(), 
                                                  left_on=["identifier", "timestep"], 
                                                  right_on=["identifier", "timestep"])
    biomass_pools = ['SoftwoodMerch','SoftwoodFoliage', 'SoftwoodOther', 'SoftwoodCoarseRoots','SoftwoodFineRoots',                        
                     'HardwoodMerch', 'HardwoodFoliage', 'HardwoodOther', 'HardwoodCoarseRoots', 'HardwoodFineRoots']
    dom_pools = ['AboveGroundVeryFastSoil', 'BelowGroundVeryFastSoil', 'AboveGroundFastSoil', 'BelowGroundFastSoil',
                 'MediumSoil', 'AboveGroundSlowSoil', 'BelowGroundSlowSoil', 'SoftwoodStemSnag', 'SoftwoodBranchSnag',
                 'HardwoodStemSnag', 'HardwoodBranchSnag']
    biomass_result = pi[['timestep']+biomass_pools]
    dom_result = pi[['timestep']+dom_pools]
    total_eco_result = pi[['timestep']+biomass_pools+dom_pools]
    annual_carbon_stocks = pd.DataFrame({'Year':pi['timestep'],
                                         'Biomass':pi[biomass_pools].sum(axis=1),
                                         'DOM':pi[dom_pools].sum(axis=1),
                                         'Total Ecosystem': pi[biomass_pools+dom_pools].sum(axis=1)})
    annual_carbon_stocks = annual_carbon_stocks.groupby('Year').sum()
    df_carbon_stock = df_carbon_stock.groupby('period').sum()    
    annual_carbon_stocks['HWP'] = df_carbon_stock['co2_stock']        
    annual_carbon_stocks['Total Ecosystem'] += df_carbon_stock['co2_stock']
    if plot:
        fig, axes = plt.subplots(2, 1, sharex=True,  figsize=(8, 8))
        annual_carbon_stocks.groupby('Year').sum().plot(ax=axes[0], xlim=(0, n_steps), ylim=(0, None))
        axes[0].set_title('Carbon stocks over years')
        axes[0].set_xlabel('Year')
        axes[0].set_ylabel('Carbon stocks')
        # plt.show()
    fi = cbm_output.classifiers.to_pandas().merge(cbm_output.flux.to_pandas(), 
                                                  left_on=["identifier", "timestep"], 
                                                  right_on=["identifier", "timestep"])    
    ecosystem_decay_emissions_pools = [
        'DecayVFastAGToAir',
        'DecayVFastBGToAir',
        'DecayFastAGToAir',
        'DecayFastBGToAir',
        'DecayMediumToAir',
        'DecaySlowAGToAir',
        'DecaySlowBGToAir',
        'DecaySWStemSnagToAir',
        'DecaySWBranchSnagToAir',
        'DecayHWStemSnagToAir',
        'DecayHWBranchSnagToAir']
    GrossGrowth_pools = [
        'DeltaBiomass_AG',
        'TurnoverMerchLitterInput',
        'TurnoverFolLitterInput',
        'TurnoverOthLitterInput',
        'DeltaBiomass_BG',
        'TurnoverCoarseLitterInput',
        'TurnoverFineLitterInput']
    ecosystem_decay_emissions_result = fi[['timestep']+ecosystem_decay_emissions_pools]
    GrossGrowth_result = fi[['timestep']+GrossGrowth_pools]
    net_emission_result = fi[['timestep']+ecosystem_decay_emissions_pools+GrossGrowth_pools]
    annual_net_emission = pd.DataFrame({ "Year": fi["timestep"],
                                        "Ecosystem decay emission": 44/12 * fi[ecosystem_decay_emissions_pools].sum(axis=1),
                                        "Gross growth": 44/12 * -1*fi[GrossGrowth_pools].sum(axis=1),
                                        "Net emission": 44/12 * (fi[ecosystem_decay_emissions_pools].sum(axis=1)-fi[GrossGrowth_pools].sum(axis=1))})
    annual_net_emission = annual_net_emission.groupby('Year').sum()
    
    df_carbon_emission =  df_carbon_emission.groupby('period').sum()
    df_carbon_emission_immed =  df_carbon_emission_immed.groupby('period').sum()
    
    df_emission_concrete_manu =  -1 * df_emission_concrete_manu.groupby('period').sum()
    df_emission_concrete_landfill =  -1 * df_emission_concrete_landfill.groupby('period').sum()
    annual_net_emission['HWP'] =  (1 - release_immediately_value) * df_carbon_emission['co2_emission'] 
    annual_net_emission['Carbon release immediately'] = release_immediately_value * df_carbon_emission_immed['co2_emission_immed']
    annual_net_emission['Concrete_manufacturing'] = df_emission_concrete_manu['co2_concrete_manu']
    annual_net_emission['Concrete_landfill'] = df_emission_concrete_landfill['co2_concrete_landfill']
    annual_net_emission['Net emission'] += annual_net_emission['HWP']
    annual_net_emission['Net emission'] += annual_net_emission['Carbon release immediately'] 
    annual_net_emission['Net emission'] += annual_net_emission['Concrete_manufacturing']
    annual_net_emission['Net emission'] += annual_net_emission['Concrete_landfill']
    if plot:
        annual_net_emission.groupby('Year').sum().plot(ax=axes[1], xlim = (0, n_steps)).axhline(y=0, color='red', linestyle='--') 
        axes[1].axhline(y=0, color='red', linestyle='--')
        axes[1].set_title('Carbon emission over years')
        axes[1].set_xlabel('Year')
        axes[1].set_ylabel('Carbon emission')
    return annual_carbon_stocks, annual_net_emission


def stock_emission_scenario(fm, clt_percentage, credibility, budget_input, n_steps, scenario_name, displacement_effect, hwp_pool_effect_value, release_immediately_value, case_study, obj_mode):   
    decay_rates = {'plumber':math.log(2.)/35., 'ppaper':math.log(2.)/2., 'pclt': math.log(2.)/float('inf')}
    product_coefficients = {'plumber': (1-0.2) * (1 - clt_percentage), 'ppaper':0.2, 'pclt': (1-0.2)*clt_percentage}
    products = ['plumber', 'ppaper', 'pclt']
    clt_conversion_rate = 1.24 #convert 1.21 lumber to 1 CLT
    co2_concrete_manu_factor = 298.
    concrete_density = 2400 #kg/m3
    co2_concrete_landfill_factor = 0.00517 * concrete_density
    sch_alt_scenario = run_scenario(fm, clt_percentage, hwp_pool_effect_value, displacement_effect, release_immediately_value, case_study, obj_mode, scenario_name, solver='gurobi')
    # sch_alt_scenario = run_scenario(fm, clt_percentage, hwp_pool_effect_value, displacement_effect, release_immediately_value, case_study, obj_mode, scenario_name) #This uses pulp

    # df = compile_scenario(fm, case_study, obj_mode, scenario_name)
    # plot_scenario(df, case_study, obj_mode, scenario_name)
    df_carbon_stock = hwp_carbon_stock(fm, products, product_coefficients, decay_rates, hwp_pool_effect_value)
    df_carbon_emission = hwp_carbon_emission(fm, products, product_coefficients, decay_rates, hwp_pool_effect_value)
    df_carbon_emission_immed = hwp_carbon_emission_immed(fm)

    df_emission_concrete_manu = emission_concrete_manu(fm, product_coefficients, credibility, clt_conversion_rate, co2_concrete_manu_factor, displacement_effect)
    df_emission_concrete_landfill = emission_concrete_landfill(fm, product_coefficients, credibility, clt_conversion_rate, co2_concrete_landfill_factor, displacement_effect)
    disturbance_type_mapping = [{'user_dist_type': 'harvest', 'default_dist_type': 'Clearcut harvesting without salvage'},
                            {'user_dist_type': 'fire', 'default_dist_type': 'Wildfire'}]
    for dtype_key in fm.dtypes:
        fm.dt(dtype_key).last_pass_disturbance = 'fire' if dtype_key[5] == dtype_key[4] else 'harvest'
    sit_config, sit_tables = fm.to_cbm_sit(softwood_volume_yname='swdvol', 
                                       hardwood_volume_yname='hwdvol', 
                                       admin_boundary='British Columbia', 
                                       eco_boundary='Montane Cordillera',
                                       disturbance_type_mapping=disturbance_type_mapping)
    cbm_output_1, cbm_output_2 = run_cbm(df_carbon_stock, df_carbon_emission,  df_carbon_emission_immed, df_emission_concrete_manu, df_emission_concrete_landfill, sit_config, sit_tables, n_steps, release_immediately_value, plot = False)
    return cbm_output_1, cbm_output_2     


def stock_emission_scenario_equivalent(fm, clt_percentage, credibility, budget_input, n_steps, max_harvest, displacement_effect, hwp_pool_effect_value, release_immediately_value, case_study, obj_mode):   
    decay_rates = {'plumber':math.log(2.)/35., 'ppaper':math.log(2.)/2.}
    product_coefficients = {'plumber':0.8, 'ppaper':0.2}
    products = ['plumber', 'ppaper', 'pclt']
    clt_conversion_rate = 1.24  #convert 1.21 lumber to 1 CLT
    co2_concrete_manu_factor = 298.
    concrete_density = 2400 #kg/m3
    co2_concrete_landfill_factor = 0.00517 * concrete_density
    sch_base_scenari = schedule_harvest_areacontrol(fm, max_harvest) #equivalent harvesting with heuristics
    df = compile_scenario(fm, case_study, obj_mode, scenario_name)
    plot_scenario(df, case_study, obj_mode, scenario_name)
    df_carbon_stock = hwp_carbon_stock(fm, products, product_coefficients, decay_rates, hwp_pool_effect_value)
    df_carbon_emission = hwp_carbon_emission(fm, products, product_coefficients, decay_rates, hwp_pool_effect_value)
    df_carbon_emission_immed = hwp_carbon_emission_immed(fm)
    df_emission_concrete_manu = emission_concrete_manu(fm, product_coefficients, credibility, clt_conversion_rate, co2_concrete_manu_factor, displacement_effect)
    df_emission_concrete_landfill = emission_concrete_landfill(fm, product_coefficients, credibility, clt_conversion_rate, co2_concrete_landfill_factor, displacement_effect)
    disturbance_type_mapping = [{'user_dist_type': 'harvest', 'default_dist_type': 'Clearcut harvesting without salvage'},
                            {'user_dist_type': 'fire', 'default_dist_type': 'Wildfire'}]
    for dtype_key in fm.dtypes:
        fm.dt(dtype_key).last_pass_disturbance = 'fire' if dtype_key[5] == dtype_key[4] else 'harvest'
    sit_config, sit_tables = fm.to_cbm_sit(softwood_volume_yname='swdvol', 
                                       hardwood_volume_yname='hwdvol', 
                                       admin_boundary='British Columbia', 
                                       eco_boundary='Montane Cordillera',
                                       disturbance_type_mapping=disturbance_type_mapping)
    cbm_output_3, cbm_output_4 = run_cbm(df_carbon_stock, df_carbon_emission,  df_carbon_emission_immed, df_emission_concrete_manu, df_emission_concrete_landfill, sit_config, sit_tables, n_steps, release_immediately_value, plot = False)
    return cbm_output_3, cbm_output_4     


def plot_scenarios(cbm_output_1, cbm_output_2, cbm_output_3, cbm_output_4, n_steps, case_study, obj_mode):
    fig_folder_path = os.path.join('./outputs/fig', case_study)
    if not os.path.exists(fig_folder_path):
        os.makedirs(fig_folder_path)
        
    output_filename = f"{case_study}_{obj_mode}_Carbon_emissions_stocks.pdf"
    output_file_path = os.path.join(fig_folder_path, output_filename)
    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(12, 10))   
    cbm_output_1.groupby('Year').sum().plot(ax=axes[0, 0], xlim=(0, n_steps), ylim=(0, None))
    axes[0, 0].set_title('Carbon stocks over years (alternative scenario)')
    axes[0, 0].set_xlabel('Year')
    axes[0, 0].set_ylabel('Carbon stocks')   
    cbm_output_2.groupby('Year').sum().plot(ax=axes[1, 0], xlim=(0, n_steps))
    axes[1, 0].axhline(y=0, color='red', linestyle='--')
    axes[1, 0].set_title('Carbon emission over years (alternative scenario)')
    axes[1, 0].set_xlabel('Year')
    axes[1, 0].set_ylabel('Carbon emission')
    cbm_output_3.groupby('Year').sum().plot(ax=axes[0, 1], xlim=(0, n_steps), ylim=(0, None))
    axes[0, 1].set_title('Carbon stocks over years (base scenario)')
    axes[0, 1].set_xlabel('Year')
    axes[0, 1].set_ylabel('Carbon stocks')    
    cbm_output_4.groupby('Year').sum().plot(ax=axes[1, 1], xlim=(0, n_steps))
    axes[1, 1].axhline(y=0, color='red', linestyle='--')
    axes[1, 1].set_title('Carbon emission over years (base scenario)')
    axes[1, 1].set_xlabel('Year')
    axes[1, 1].set_ylabel('Carbon emission')    
    plt.tight_layout()
    plt.savefig(output_file_path)
    plt.show()
    


def scenario_dif(cbm_output_2, cbm_output_4, budget_input, n_steps, case_study, obj_mode):
    
    fig_folder_path = os.path.join('./outputs/fig', case_study)
    if not os.path.exists(fig_folder_path):
        os.makedirs(fig_folder_path)
        
    output_filename = f"{case_study}_{obj_mode}_net_emission_difference.pdf"
    output_file_path = os.path.join(fig_folder_path, output_filename)
    cbm_output_2.reset_index(drop=False, inplace=True)
    dif_scenario = pd.DataFrame({"Year": cbm_output_2["Year"],
                       "Net emission": cbm_output_2['Net emission'] - cbm_output_4['Net emission']})
    ax = dif_scenario.groupby('Year').sum().plot(xlim = (0, n_steps))
    ax.axhline(y=0, color='red', linestyle='--')
    ax.set_title('Net emission difference between base and alternative scenarios')
    ax.set_xlabel('Year')
    ax.set_ylabel('Net Carbon emission diffrence')   
    # dollar_per_ton = abs(budget_input / dif_scenario.iloc[:25]['Net emission'].sum()) # Calculate for the next 25 years
    # print( "Net emission difference", dif_scenario.iloc[:25]['Net emission'].sum())
    # print( "Net emission base scenario", cbm_output_4.iloc[:25]['Net emission'].sum())
    # print( "Net emission alternative scenario", cbm_output_2.iloc[:25]['Net emission'].sum())    

    dollar_per_ton = abs(budget_input / dif_scenario['Net emission'].sum()) # Calculate for the next 25 years
    print( "Net emission difference", dif_scenario['Net emission'].sum())
    print( "Net emission base scenario", cbm_output_4['Net emission'].sum())
    print( "Net emission alternative scenario", cbm_output_2['Net emission'].sum()) 
    print('dollar_per_ton is: ', dollar_per_ton)
    plt.savefig(output_file_path)
    return ax



def compare_kpi_age(kpi_age_base, kpi_age_alt, case_study, obj_mode):
    import os  
    
    # Calculate the difference in old growth area between the two scenarios
    comparison_df = kpi_age_alt - kpi_age_base
    comparison_df['Difference'] = comparison_df[10] - comparison_df[0]
    
    print("Comparison of Old Growth Areas (Alternative - Base)")
    print(comparison_df)
    
    total_difference = comparison_df['Difference'].sum()
    if total_difference < 0:
        print(f"\nOverall, the old growth area has **decreased** by {total_difference:.2f} hectares in the alternative scenario compared to the base scenario.")
    else:
        print(f"\nOverall, the old growth area has **increased** by {total_difference:.2f} hectares in the alternative scenario compared to the base scenario.")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    comparison_df[['Difference']].plot(kind='bar', color="#FF4500", ax=ax)
    
    ax.set_title("Difference in Old Growth Area by Species (Alternative vs Base)")
    ax.set_xlabel("Species")
    ax.set_ylabel("Difference in Old Growth Area (ha)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    folder_path = os.path.join('./outputs/fig', case_study)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)   
    file_name = f"{case_study}_{obj_mode}_kpi_age_difference.pdf"
    file_path = os.path.join(folder_path, file_name)  
    plt.savefig(file_path)
    plt.show()
    plt.close()   
    print(f"Plot saved to {file_path}")

    return comparison_df





def compare_kpi_species(portion_10_alt , shannon_10_alt, portion_10_base, shannon_10_base, case_study, obj_mode):
    
    import matplotlib.pyplot as plt
    import os

    colors = {
        'Aspen': '#FF0000', 'Bal': '#FF8C00', 'Cedar': '#FFD700', 'Alder': '#00FF00',
        'DougFir': '#00FFFF', 'Hem': '#1E90FF', 'Pine': '#9400D3', 'Spruce': '#FF00FF'
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    plt.subplots_adjust(wspace=0.6)

    # Pie chart for time period 10 for tohe alternative scenario
    axes[0].pie(portion_10_alt.values(), labels=portion_10_alt.keys(), 
                colors=[colors[species] for species in portion_10_alt.keys()], autopct='%1.1f%%', startangle=140)
    axes[0].set_title("Species Distribution (Time Period 10, Alternative Scenario)")

    # Pie chart for time period 10 for the base scenario
    axes[1].pie(portion_10_base.values(), labels=portion_10_base.keys(), 
                colors=[colors[species] for species in portion_10_base.keys()], autopct='%1.1f%%', startangle=140)
    axes[1].set_title("Species Distribution (Time Period 10, Base Scenario)")

    fig.suptitle(f"Shannon Index for Time Period 10: Alternative Scenario: {shannon_10_alt:.4f}, Base Scenario: {shannon_10_base:.4f}", y=0.05, fontsize=12)

    folder_path = os.path.join('./outputs/fig', case_study)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    file_name = f"{case_study}_{obj_mode}_kpi_species_difference_pie.pdf"
    file_path = os.path.join(folder_path, file_name)
    plt.savefig(file_path)
    plt.show()
    plt.close()
    
    print(f"Pie Charts for Time Periods 0 and 10 saved to {file_path}")
   


def compare_kpi_socioeconomic(kpi_socio_alt, kpi_eco_alt, kpi_socio_base, kpi_eco_base):
    """
    Compare the socioeconomic KPIs between the alternative and baseline scenarios.
    """
    socio_diff = kpi_socio_alt - kpi_socio_base
    eco_diff = kpi_eco_alt - kpi_eco_base
    
    print("Comparison of Socioeconomic KPIs:")
    print("------------------------------------------------")
    print(f"Job Creation Difference (Social Indicator): {socio_diff}")
    print(f"Provincial Revenue Difference (Economic Indicator): {eco_diff}")
    print("------------------------------------------------")
    
    if socio_diff > 0:
        print("Alternative scenario creates more jobs than the baseline.")
    elif socio_diff < 0:
        print("Baseline scenario creates more jobs than the alternative.")
    else:
        print("Both scenarios create the same number of jobs.")
        
    if eco_diff > 0:
        print("Alternative scenario generates more provincial revenue than the baseline.")
    elif eco_diff < 0:
        print("Baseline scenario generates more provincial revenue than the alternative.")
    else:
        print("Both scenarios generate the same amount of provincial revenue.")
    
    return socio_diff, eco_diff
    
    


def results_scenarios(fm, clt_percentage, credibility, budget_input, n_steps, max_harvest, scenario_name, displacement_effect, hwp_pool_effect_value, release_immediately_value, case_study, obj_mode, pickle_output_base,  
                  pickle_output_alter):
    from util import stock_emission_scenario, plot_scenarios, scenario_dif, stock_emission_scenario_equivalent, compare_kpi_age, kpi_socioeconomic, compare_kpi_socioeconomic


    # Create a folder for pickle outputs
    pickle_folder_path = os.path.join('./outputs/pickle', case_study)
    if not os.path.exists(pickle_folder_path):
        os.makedirs(pickle_folder_path)

     # Run Alternative scenario
    # Define pickle file paths_alter
    pickle_file_1 = os.path.join(pickle_folder_path, f'{case_study}_{obj_mode}_{scenario_name}_cbm_output_1.pkl')
    pickle_file_2 = os.path.join(pickle_folder_path, f'{case_study}_{obj_mode}_{scenario_name}_cbm_output_2.pkl')

   # Check if pickled cbm_output_1 and cbm_output_2 already exist
    if pickle_output_alter and os.path.exists(pickle_file_1) and os.path.exists(pickle_file_2):
        # Load pickle files if they exist
        with open(pickle_file_1, 'rb') as f:
            cbm_output_1 = pickle.load(f)
        with open(pickle_file_2, 'rb') as f:
            cbm_output_2 = pickle.load(f)
        print("Loaded cbm_output_1 and cbm_output_2 from pickle files.")
    else:
        cbm_output_1, cbm_output_2 = stock_emission_scenario(fm, clt_percentage, credibility, budget_input, n_steps, scenario_name, displacement_effect, hwp_pool_effect_value, release_immediately_value, case_study, obj_mode)
        with open(pickle_file_1, 'wb') as f:
            pickle.dump(cbm_output_1, f)
        with open(pickle_file_2, 'wb') as f:
            pickle.dump(cbm_output_2, f)
        print("Saved cbm_output_1 and cbm_output_2 as pickle files.")
    
    kpi_age_alt = kpi_age(fm, case_study, obj_mode, scenario_name)
    portion_10_alt , shannon_10_alt = kpi_species(fm, case_study, obj_mode, scenario_name)
    # kpi_socio_alt, kpi_eco_alt = kpi_socioeconomic(fm)

    # Create a folder for csv file outputs
    csv_folder_path = os.path.join('./outputs/csv', case_study)
    if not os.path.exists(csv_folder_path):
        os.makedirs(csv_folder_path)
    # Save cbm_output_2 as CSV
    cbm_output_2_df = pd.DataFrame(cbm_output_2)
    cbm_output_2_file = os.path.join(csv_folder_path, f'{case_study}_{obj_mode}_{scenario_name}_cbm_output_2.csv')
    cbm_output_2_df.to_csv(cbm_output_2_file, index=False)
    # print(cbm_output_2)
    cbm_output_1_df = pd.DataFrame(cbm_output_1)
    cbm_output_1_file = os.path.join(csv_folder_path, f'{case_study}_{obj_mode}_{scenario_name}_cbm_output_1.csv')
    cbm_output_1_df.to_csv(cbm_output_1_file, index=False)

    fm.reset()

    # Run Base scenario
    # Define pickle file paths_base
    pickle_file_3 = os.path.join(pickle_folder_path, f'{case_study}_{obj_mode}_{scenario_name}_cbm_output_3.pkl')
    pickle_file_4 = os.path.join(pickle_folder_path, f'{case_study}_{obj_mode}_{scenario_name}_cbm_output_4.pkl')

    # Check if pickled cbm_output_3 and cbm_output_4 already exist
    if pickle_output_base and os.path.exists(pickle_file_3) and os.path.exists(pickle_file_4):
        # Load pickle files if they exist
        with open(pickle_file_3, 'rb') as f:
            cbm_output_3 = pickle.load(f)
        with open(pickle_file_4, 'rb') as f:
            cbm_output_4 = pickle.load(f)
        print("Loaded cbm_output_3 and cbm_output_4 from pickle files.")
    else:
        # Run base scenario if pickle files don't exist
        if case_study == 'mining_site_1':
            scenario_name = 'bau_ms1'
        elif case_study == 'mining_site_2':
            scenario_name = 'bau_ms2'
        elif case_study == 'mining_site_3':
            scenario_name = 'bau_ms3'
        elif case_study == 'test':
            scenario_name = 'no_cons'
        else:
            raise ValueError('Invalid case_study: %s' % case_study)

        cbm_output_3, cbm_output_4 = stock_emission_scenario(fm, clt_percentage, credibility, budget_input, n_steps, scenario_name, displacement_effect, hwp_pool_effect_value, release_immediately_value, case_study, obj_mode)
        
        # Save cbm_output_3 and cbm_output_4 as pickle
        with open(pickle_file_3, 'wb') as f:
            pickle.dump(cbm_output_3, f)
        with open(pickle_file_4, 'wb') as f:
            pickle.dump(cbm_output_4, f)
        print("Saved cbm_output_3 and cbm_output_4 as pickle files.")

    kpi_age_base = kpi_age(fm, case_study, obj_mode, scenario_name)
    portion_10_base , shannon_10_base = kpi_species(fm, case_study, obj_mode, scenario_name)
    # kpi_socio_base, kpi_eco_base = kpi_socioeconomic(fm)

    # Save cbm_output_4 as CSV
    cbm_output_4_df = pd.DataFrame(cbm_output_4)
    cbm_output_4_file = os.path.join(csv_folder_path, f'{case_study}_{obj_mode}_{scenario_name}_cbm_output_4.csv')
    cbm_output_4_df.to_csv(cbm_output_4_file, index=False)
    # print(cbm_output_4)
    cbm_output_3_df = pd.DataFrame(cbm_output_3)
    cbm_output_3_file = os.path.join(csv_folder_path, f'{case_study}_{obj_mode}_{scenario_name}_cbm_output_3.csv')
    cbm_output_3_df.to_csv(cbm_output_3_file, index=False)

    # Plot scenarios
    plot_scenarios(cbm_output_1, cbm_output_2, cbm_output_3, cbm_output_4, n_steps, case_study, obj_mode)
    
    # Scenario difference plot
    print("---------------------------------------------------------------------------------------")
    dif_plot = scenario_dif(cbm_output_2, cbm_output_4, budget_input, n_steps, case_study, obj_mode)

    compare_kpi_age(kpi_age_base, kpi_age_alt, case_study, obj_mode)

    compare_kpi_species(portion_10_alt , shannon_10_alt, portion_10_base, shannon_10_base, case_study, obj_mode)

    # compare_kpi_socioeconomic(kpi_socio_alt, kpi_eco_alt, kpi_socio_base, kpi_eco_base)
    print("---------------------------------------------------------------------------------------")






def cbm_report(fm, cbm_output, biomass_pools, dom_pools, fluxes, gross_growth):
    # Add carbon pools indicators 
    pi = cbm_output.classifiers.to_pandas().merge(cbm_output.pools.to_pandas(), 
                                                  left_on=["identifier", "timestep"], 
                                                  right_on=["identifier", "timestep"])

    annual_carbon_stock = pd.DataFrame({'Year': pi['timestep'],
                                         'Biomass': pi[biomass_pools].sum(axis=1),
                                         'DOM': pi[dom_pools].sum(axis=1),
                                         'Ecosystem': pi[biomass_pools + dom_pools].sum(axis=1)})
    
    annual_product_stock = pd.DataFrame({'Year': pi['timestep'],
                                         'Product': pi['Products']})
    
    annual_stock_change = annual_carbon_stock[['Year', 'Ecosystem']].copy()
    annual_stock_change['Stock_Change'] = annual_stock_change['Ecosystem'].diff()
    annual_stock_change = annual_stock_change[['Year', 'Stock_Change']]
    annual_stock_change.loc[annual_stock_change['Year'] == 0, 'Stock_Change'] = 0
     
    fi = cbm_output.classifiers.to_pandas().merge(cbm_output.flux.to_pandas(), 
                                                  left_on=["identifier", "timestep"], 
                                                  right_on=["identifier", "timestep"])
    
    annual_all_emission = pd.DataFrame({'Year': fi['timestep'],
                                         'All_Emissions': fi[fluxes].sum(axis=1)})
    
    annual_gross_growth = pd.DataFrame({'Year': fi['timestep'],
                                        'Gross_Growth': fi[gross_growth].sum(axis=1)})
     
    n_steps = fm.horizon * fm.period_length
    annual_carbon_stock.groupby('Year').sum().plot(
        figsize=(5, 5), xlim=(0, n_steps), ylim=(None, None), xlabel="Year", ylabel="Stock (ton C)",
        title="Annual Carbon Stock"
    )

    annual_all_emission.groupby('Year').sum().plot(
        figsize=(5, 5), xlim=(0, n_steps), ylim=(None, None),
        title="Annual Ecosystem Carbon Emission", xlabel="Year", ylabel="Stock (ton C)"
    )

    annual_stock_change.groupby('Year').sum().plot(
        figsize=(5, 5), xlim=(0, n_steps), ylim=(None, None),
        title="Annual Ecosystem Carbon Stock Change", xlabel="Year", ylabel="tons of C"
    )

    annual_gross_growth.groupby('Year').sum().plot(
        figsize=(5, 5), xlim=(0, n_steps), ylim=(None, None),
        title="Annual Forest Gross Growth", xlabel="Year", ylabel="tons of C"
    )

    df_cs = annual_carbon_stock.groupby('Year').sum()
    df_ae = annual_all_emission.groupby('Year').sum()
    df_gg = annual_gross_growth.groupby('Year').sum()
    df_sc = annual_stock_change.groupby('Year').sum()

    # Correctly merging all dataframes
    merged_df = pd.merge(pd.merge(pd.merge(df_cs, df_ae, left_index=True, right_index=True, how='outer'),
                                  df_gg, left_index=True, right_index=True, how='outer'),
                         df_sc, left_index=True, right_index=True, how='outer')

    merged_df['Stock_Change'] = merged_df['Ecosystem'].diff() * (-1)
    merged_df.at[0, 'Stock_Change'] = 0

    return merged_df


def cbm_report_both(fm, cbm_output, biomass_pools, dom_pools, fluxes, gross_growth):
    # Add carbon pools indicators 
    pi = cbm_output.classifiers.to_pandas().merge(cbm_output.pools.to_pandas(), 
                                                  left_on=["identifier", "timestep"], 
                                                  right_on=["identifier", "timestep"])

    annual_carbon_stock = pd.DataFrame({'Year': pi['timestep'],
                                         'Biomass': pi[biomass_pools].sum(axis=1),
                                         'DOM': pi[dom_pools].sum(axis=1),
                                         'Ecosystem': pi[biomass_pools + dom_pools].sum(axis=1)})
    
    annual_product_stock = pd.DataFrame({'Year': pi['timestep'],
                                         'Product': pi['Products']})
    
    annual_stock_change = annual_carbon_stock[['Year', 'Ecosystem']].copy()
    annual_stock_change['Stock_Change'] = annual_stock_change['Ecosystem'].diff()
    annual_stock_change = annual_stock_change[['Year', 'Stock_Change']]
    annual_stock_change.loc[annual_stock_change['Year'] == 0, 'Stock_Change'] = 0
     
    fi = cbm_output.classifiers.to_pandas().merge(cbm_output.flux.to_pandas(), 
                                                  left_on=["identifier", "timestep"], 
                                                  right_on=["identifier", "timestep"])
    
    annual_all_emission = pd.DataFrame({'Year': fi['timestep'],
                                         'All_Emissions': 44/12 * fi[fluxes].sum(axis=1)})
    
    annual_gross_growth = pd.DataFrame({'Year': fi['timestep'],
                                        'Gross_Growth': 44/12 * -1 * fi[gross_growth].sum(axis=1)})

    annual_net_emission = pd.DataFrame({'Year': fi['timestep'],
                                        'Net_Emission': 44/12 * (fi[fluxes].sum(axis=1) - fi[gross_growth].sum(axis=1)) })

    
    n_steps = fm.horizon * fm.period_length
    annual_carbon_stock.groupby('Year').sum().plot(
        figsize=(5, 5), xlim=(0, n_steps), ylim=(None, None), xlabel="Year", ylabel="Stock (ton C)",
        title="Annual Carbon Stock"
    )

    annual_all_emission.groupby('Year').sum().plot(
        figsize=(5, 5), xlim=(0, n_steps), ylim=(None, None),
        title="Annual Ecosystem Carbon Emission", xlabel="Year", ylabel="Stock (ton C)"
    )

    annual_stock_change.groupby('Year').sum().plot(
        figsize=(5, 5), xlim=(0, n_steps), ylim=(None, None),
        title="Annual Ecosystem Carbon Stock Change", xlabel="Year", ylabel="tons of C"
    )

    annual_gross_growth.groupby('Year').sum().plot(
        figsize=(5, 5), xlim=(0, n_steps), ylim=(None, None),
        title="Annual Forest Gross Growth", xlabel="Year", ylabel="tons of C"
    )

    df_cs = annual_carbon_stock.groupby('Year').sum()
    df_ae = annual_all_emission.groupby('Year').sum()
    df_gg = annual_gross_growth.groupby('Year').sum()
    df_sc = annual_stock_change.groupby('Year').sum()
    df_ne = annual_net_emission.groupby('Year').sum()

    # Correctly merging all dataframes
    # merged_df = pd.merge(pd.merge(pd.merge(df_cs, df_ae, left_index=True, right_index=True, how='outer'),
    #                               df_gg, left_index=True, right_index=True, how='outer'),
    #                      df_sc, left_index=True, right_index=True, how='outer')

    merged_df = pd.concat([df_cs, df_ae, df_gg, df_sc, df_ne], axis=1)

    merged_df['Stock_Change'] = merged_df['Ecosystem'].diff() * (-1)
    merged_df.at[0, 'Stock_Change'] = 0

    return merged_df




    

def compare_ws3_cbm(fm, cbm_output, disturbance_type_mapping, biomass_pools, dom_pools, plots):
    import numpy as np
    eco_pools = biomass_pools + dom_pools
    pi = cbm_output.classifiers.to_pandas().merge(cbm_output.pools.to_pandas(), 
                                                  left_on=["identifier", "timestep"], 
                                                  right_on=["identifier", "timestep"])

    df_cbm = pd.DataFrame({'period': pi["timestep"] * 0.1, 
                       'biomass_stock': pi[biomass_pools].sum(axis=1),
                       'dom_stock': pi[dom_pools].sum(axis=1),
                       'eco_stock': pi[eco_pools].sum(axis=1)}).groupby('period').sum().iloc[1::10, :].reset_index()
    df_cbm['period'] = (df_cbm['period'] + 0.9).astype(int)

    df_cbm['eco_stock_change'] = df_cbm['eco_stock'].diff()
    df_cbm.at[0, 'eco_stock_change'] = 0.

    df_ws3 = pd.DataFrame({'period': fm.periods,
                           'biomass_stock': [fm.inventory(period, 'biomass') for period in fm.periods],
                           'dom_stock': [fm.inventory(period, 'DOM') for period in fm.periods],
                           'eco_stock': [fm.inventory(period, 'ecosystem') for period in fm.periods]})
  

    df_ws3['eco_stock_change'] = df_ws3['eco_stock'].diff()
    df_ws3.at[0, 'eco_stock_change'] = 0.

    if plots == "whole":
        # Create a figure for all comparisons in one plot
        plt.figure(figsize=(10, 6))
    
        # Plotting the ecosystem stock comparison
        plt.plot(df_cbm['period'], df_cbm['eco_stock'], label='CBM Ecosystem Stock')
        plt.plot(df_ws3['period'], df_ws3['eco_stock'], label='WS3 Ecosystem Stock')
    
        # Plotting the biomass stock comparison
        plt.plot(df_cbm['period'], df_cbm['biomass_stock'], label='CBM Biomass Stock')
        plt.plot(df_ws3['period'], df_ws3['biomass_stock'], label='WS3 Biomass Stock')
    
        # Plotting the DOM stock comparison
        plt.plot(df_cbm['period'], df_cbm['dom_stock'], label='CBM DOM Stock')
        plt.plot(df_ws3['period'], df_ws3['dom_stock'], label='WS3 DOM Stock')
    
        # Set labels and title
        plt.xlabel('Period')
        plt.ylabel('Stock (ton C)')
        plt.ylim(0, None)  # Ensure y-axis starts from 0
    
        # Customize x-axis ticks to show every 2 periods
        ticks = np.arange(df_cbm['period'].min()-1, df_cbm['period'].max() + 1, 2)
        plt.xticks(ticks)
        
        # Add a legend to differentiate the lines
        plt.legend()

    if plots == "individual":
        # Create a figure with subplots
        fig, axs = plt.subplots(3, 1, figsize=(8, 12))
        
        # Define x-axis ticks (0 to 20 with a step of 2)
        ticks = np.arange(df_cbm['period'].min()-1, df_cbm['period'].max() + 1, 2)
        
        # Plotting the ecosystem stock comparison
        axs[0].plot(df_cbm['period'], df_cbm['eco_stock'], label='cbm ecosystem stock')
        axs[0].plot(df_ws3['period'], df_ws3['eco_stock'], label='ws3 ecosystem stock')
        axs[0].set_xlabel('Period')
        axs[0].set_ylabel('Stock (ton C)')
        # axs[0].set_ylim(0, None)  # Set y-axis to start from 0
        axs[0].set_xticks(ticks)  # Set x-axis ticks to show every 2 periods
        axs[0].legend()
        
        # Plotting the biomass stock comparison
        axs[1].plot(df_cbm['period'], df_cbm['biomass_stock'], label='cbm biomass stock')
        axs[1].plot(df_ws3['period'], df_ws3['biomass_stock'], label='ws3 biomass stock')
        axs[1].set_xlabel('Period')
        axs[1].set_ylabel('Stock (ton C)')
        # axs[1].set_ylim(0, None)  # Set y-axis to start from 0
        axs[1].set_xticks(ticks)  # Set x-axis ticks to show every 2 periods
        axs[1].legend()
        
        # Plotting the DOM stock comparison
        axs[2].plot(df_cbm['period'], df_cbm['dom_stock'], label='cbm dom stock')
        axs[2].plot(df_ws3['period'], df_ws3['dom_stock'], label='ws3 dom stock')
        axs[2].set_xlabel('Period')
        axs[2].set_ylabel('Stock (ton C)')
        # axs[2].set_ylim(0, None)  # Set y-axis to start from 0
        axs[2].set_xticks(ticks)  # Set x-axis ticks to show every 2 periods
        axs[2].legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the combined plot
    plt.show()

    return df_cbm, df_ws3


def compare_ws3_cbm_both(fm, cbm_output, disturbance_type_mapping, biomass_pools, dom_pools, ecosystem_decay_emissions_pools, GrossGrowth_pools, plots):
    import numpy as np
    eco_pools = biomass_pools + dom_pools
    pi = cbm_output.classifiers.to_pandas().merge(cbm_output.pools.to_pandas(), 
                                                  left_on=["identifier", "timestep"], 
                                                  right_on=["identifier", "timestep"])
    fi = cbm_output.classifiers.to_pandas().merge(cbm_output.flux.to_pandas(), 
                                                  left_on=["identifier", "timestep"], 
                                                  right_on=["identifier", "timestep"])

    df_cbm = pd.DataFrame({'period': pi["timestep"] * 0.1, 
                       'biomass_stock': pi[biomass_pools].sum(axis=1),
                       'dom_stock': pi[dom_pools].sum(axis=1),
                       'eco_stock': pi[eco_pools].sum(axis=1),
                       'ecosystem_decay_emissions': 44/12 * fi[ecosystem_decay_emissions_pools].sum(axis=1),
                       'gross_growth': 44/12 * -1* fi[GrossGrowth_pools].sum(axis=1),
                       'net_emission': 44/12 * ( fi[ecosystem_decay_emissions_pools].sum(axis=1) - fi[GrossGrowth_pools].sum(axis=1)) }).groupby('period').sum().iloc[1::10, :].reset_index()
    df_cbm['period'] = (df_cbm['period'] + 0.9).astype(int)


    df_cbm['eco_stock_change'] = df_cbm['eco_stock'].diff()
    df_cbm.at[0, 'eco_stock_change'] = 0.

    df_ws3 = pd.DataFrame({'period': fm.periods,
                           'biomass_stock': [fm.inventory(period, 'biomass') for period in fm.periods],
                           'dom_stock': [fm.inventory(period, 'DOM') for period in fm.periods],
                           'eco_stock': [fm.inventory(period, 'ecosystem') for period in fm.periods],
                          'net_emission': [fm.inventory(period, 'net_emission') for period in fm.periods]})
  

    df_ws3['eco_stock_change'] = df_ws3['eco_stock'].diff()
    df_ws3.at[0, 'eco_stock_change'] = 0.

    if plots == "whole":
        # Create a figure for all comparisons in one plot
        plt.figure(figsize=(10, 6))
    
        # Plotting the ecosystem stock comparison
        plt.plot(df_cbm['period'], df_cbm['eco_stock'], label='CBM Ecosystem Stock')
        plt.plot(df_ws3['period'], df_ws3['eco_stock'], label='WS3 Ecosystem Stock')
    
        # Plotting the biomass stock comparison
        plt.plot(df_cbm['period'], df_cbm['biomass_stock'], label='CBM Biomass Stock')
        plt.plot(df_ws3['period'], df_ws3['biomass_stock'], label='WS3 Biomass Stock')
    
        # Plotting the DOM stock comparison
        plt.plot(df_cbm['period'], df_cbm['dom_stock'], label='CBM DOM Stock')
        plt.plot(df_ws3['period'], df_ws3['dom_stock'], label='WS3 DOM Stock')

        # Plotting the Net emissions comparison
        plt.plot(df_cbm['period'], df_cbm['net_emission'], label='CBM Net Emissions')
        plt.plot(df_ws3['period'], df_ws3['net_emission'], label='WS3 Net Emissions')
        
        # Set labels and title
        plt.xlabel('Period')
        plt.ylabel('Stock (ton C)')
        plt.ylim(0, None)  # Ensure y-axis starts from 0
    
        # Customize x-axis ticks to show every 2 periods
        ticks = np.arange(df_cbm['period'].min()-1, df_cbm['period'].max() + 1, 2)
        plt.xticks(ticks)
        
        # Add a legend to differentiate the lines
        plt.legend()

    if plots == "individual":
        # Create a figure with subplots
        fig, axs = plt.subplots(4, 1, figsize=(8, 12))
        
        # Define x-axis ticks (0 to 20 with a step of 2)
        ticks = np.arange(df_cbm['period'].min()-1, df_cbm['period'].max() + 1, 2)
        
        # Plotting the ecosystem stock comparison
        axs[0].plot(df_cbm['period'], df_cbm['eco_stock'], label='cbm ecosystem stock')
        axs[0].plot(df_ws3['period'], df_ws3['eco_stock'], label='ws3 ecosystem stock')
        axs[0].set_xlabel('Period')
        axs[0].set_ylabel('Stock (ton C)')
        # axs[0].set_ylim(0, None)  # Set y-axis to start from 0
        axs[0].set_xticks(ticks)  # Set x-axis ticks to show every 2 periods
        axs[0].legend()
        
        # Plotting the biomass stock comparison
        axs[1].plot(df_cbm['period'], df_cbm['biomass_stock'], label='cbm biomass stock')
        axs[1].plot(df_ws3['period'], df_ws3['biomass_stock'], label='ws3 biomass stock')
        axs[1].set_xlabel('Period')
        axs[1].set_ylabel('Stock (ton C)')
        # axs[1].set_ylim(0, None)  # Set y-axis to start from 0
        axs[1].set_xticks(ticks)  # Set x-axis ticks to show every 2 periods
        axs[1].legend()
        
        # Plotting the DOM stock comparison
        axs[2].plot(df_cbm['period'], df_cbm['dom_stock'], label='cbm dom stock')
        axs[2].plot(df_ws3['period'], df_ws3['dom_stock'], label='ws3 dom stock')
        axs[2].set_xlabel('Period')
        axs[2].set_ylabel('Stock (ton C)')
        # axs[2].set_ylim(0, None)  # Set y-axis to start from 0
        axs[2].set_xticks(ticks)  # Set x-axis ticks to show every 2 periods
        axs[2].legend()
       
        # Plotting the DOM stock comparison
        axs[3].plot(df_cbm['period'], df_cbm['net_emission'], label='cbm net emissions')
        axs[3].plot(df_ws3['period'], df_ws3['net_emission'], label='ws3 net emissions')
        axs[3].set_xlabel('Period')
        axs[3].set_ylabel('Carbon emission')
        # axs[3].set_ylim(0, None)  # Set y-axis to start from 0
        axs[3].set_xticks(ticks)  # Set x-axis ticks to show every 2 periods
        axs[3].legend()
    
    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the combined plot
    plt.show()

    return df_cbm, df_ws3




def compare_ws3_cbm_exactmatch(fm, cbm_output, disturbance_type_mapping, biomass_pools, dom_pools, plots):
    import numpy as np
    eco_pools = biomass_pools + dom_pools
    pi = cbm_output.classifiers.to_pandas().merge(cbm_output.pools.to_pandas(), 
                                                  left_on=["identifier", "timestep"], 
                                                  right_on=["identifier", "timestep"])

    df_cbm = pd.DataFrame({'period': pi["timestep"] * 0.1, 
                       'biomass_stock': pi[biomass_pools].sum(axis=1),
                       'dom_stock': pi[dom_pools].sum(axis=1),
                       'eco_stock': pi[eco_pools].sum(axis=1)}).groupby('period').sum().iloc[1::10, :].reset_index()
    df_cbm['period'] = (df_cbm['period'] + 0.9).astype(int)

    df_cbm['eco_stock_change'] = df_cbm['eco_stock'].diff()
    df_cbm.at[0, 'eco_stock_change'] = 0.

    # df_ws3 = pd.DataFrame({'period': fm.periods,
    #                        'biomass_stock': [sum(fm.inventory(period, pool) for pool in ['biomass']) for period in fm.periods],
    #                        'dom_stock': [sum(fm.inventory(period, pool) for pool in ['DOM']) for period in fm.periods],
    #                        'eco_stock': [sum(fm.inventory(period, pool) for pool in ['ecosystem']) for period in fm.periods]})
  
    df_ws3 = pd.DataFrame({'period': fm.periods,
                           'biomass_stock': [sum(fm.inventory(period, pool)/fm.inventory(period) for pool in ['biomass']) for period in fm.periods],
                           'dom_stock': [sum(fm.inventory(period, pool)/fm.inventory(period) for pool in ['DOM']) for period in fm.periods],
                           'eco_stock': [sum(fm.inventory(period, pool)/fm.inventory(period) for pool in ['ecosystem']) for period in fm.periods]})

    df_ws3['eco_stock_change'] = df_ws3['eco_stock'].diff()
    df_ws3.at[0, 'eco_stock_change'] = 0.

    if plots == "whole":
        # Create a figure for all comparisons in one plot
        plt.figure(figsize=(10, 6))
    
        # Plotting the ecosystem stock comparison
        plt.plot(df_cbm['period'], df_cbm['eco_stock'], label='CBM Ecosystem Stock')
        plt.plot(df_ws3['period'], df_ws3['eco_stock'], label='WS3 Ecosystem Stock')
    
        # Plotting the biomass stock comparison
        plt.plot(df_cbm['period'], df_cbm['biomass_stock'], label='CBM Biomass Stock')
        plt.plot(df_ws3['period'], df_ws3['biomass_stock'], label='WS3 Biomass Stock')
    
        # Plotting the DOM stock comparison
        plt.plot(df_cbm['period'], df_cbm['dom_stock'], label='CBM DOM Stock')
        plt.plot(df_ws3['period'], df_ws3['dom_stock'], label='WS3 DOM Stock')
    
        # Set labels and title
        plt.xlabel('Period')
        plt.ylabel('Stock (ton C)')
        plt.ylim(0, None)  # Ensure y-axis starts from 0
    
        # Customize x-axis ticks to show every 2 periods
        ticks = np.arange(df_cbm['period'].min()-1, df_cbm['period'].max() + 1, 2)
        plt.xticks(ticks)
        
        # Add a legend to differentiate the lines
        plt.legend()

    if plots == "individual":
        # Create a figure with subplots
        fig, axs = plt.subplots(3, 1, figsize=(8, 12))
        
        # Define x-axis ticks (0 to 20 with a step of 2)
        ticks = np.arange(df_cbm['period'].min()-1, df_cbm['period'].max() + 1, 2)
        
        # Plotting the ecosystem stock comparison
        axs[0].plot(df_cbm['period'], df_cbm['eco_stock'], label='cbm ecosystem stock')
        axs[0].plot(df_ws3['period'], df_ws3['eco_stock'], label='ws3 ecosystem stock')
        axs[0].set_xlabel('Period')
        axs[0].set_ylabel('Stock (ton C)')
        # axs[0].set_ylim(0, None)  # Set y-axis to start from 0
        axs[0].set_xticks(ticks)  # Set x-axis ticks to show every 2 periods
        axs[0].legend()
        
        # Plotting the biomass stock comparison
        axs[1].plot(df_cbm['period'], df_cbm['biomass_stock'], label='cbm biomass stock')
        axs[1].plot(df_ws3['period'], df_ws3['biomass_stock'], label='ws3 biomass stock')
        axs[1].set_xlabel('Period')
        axs[1].set_ylabel('Stock (ton C)')
        # axs[1].set_ylim(0, None)  # Set y-axis to start from 0
        axs[1].set_xticks(ticks)  # Set x-axis ticks to show every 2 periods
        axs[1].legend()
        
        # Plotting the DOM stock comparison
        axs[2].plot(df_cbm['period'], df_cbm['dom_stock'], label='cbm dom stock')
        axs[2].plot(df_ws3['period'], df_ws3['dom_stock'], label='ws3 dom stock')
        axs[2].set_xlabel('Period')
        axs[2].set_ylabel('Stock (ton C)')
        # axs[2].set_ylim(0, None)  # Set y-axis to start from 0
        axs[2].set_xticks(ticks)  # Set x-axis ticks to show every 2 periods
        axs[2].legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the combined plot
    plt.show()

    return df_cbm, df_ws3


def plugin_c_curves_both(fm, c_curves_p, c_curves_f):
    # Define Sum Carbon Pools and Sum Carbon Fluxes
    biomass_pools = ['SoftwoodMerch','SoftwoodFoliage', 'SoftwoodOther', 'SoftwoodCoarseRoots','SoftwoodFineRoots',                        
                     'HardwoodMerch', 'HardwoodFoliage', 'HardwoodOther', 'HardwoodCoarseRoots', 'HardwoodFineRoots']
    dom_pools = ['AboveGroundVeryFastSoil', 'BelowGroundVeryFastSoil', 'AboveGroundFastSoil', 'BelowGroundFastSoil',
                 'MediumSoil', 'AboveGroundSlowSoil', 'BelowGroundSlowSoil', 'SoftwoodStemSnag', 'SoftwoodBranchSnag',
                 'HardwoodStemSnag', 'HardwoodBranchSnag']
    all_fluxes = [
        'DisturbanceCO2Production',
        'DisturbanceCH4Production',
        'DisturbanceCOProduction',
        'DisturbanceBioCO2Emission',
        'DisturbanceBioCH4Emission',
        'DisturbanceBioCOEmission',
        'DecayDOMCO2Emission',
        'DisturbanceSoftProduction',
        'DisturbanceHardProduction',
        'DisturbanceDOMProduction',
        'DeltaBiomass_AG',
        'DeltaBiomass_BG',
        'TurnoverMerchLitterInput',
        'TurnoverFolLitterInput',
        'TurnoverOthLitterInput',
        'TurnoverCoarseLitterInput',
        'TurnoverFineLitterInput',
        'DecayVFastAGToAir',
        'DecayVFastBGToAir',
        'DecayFastAGToAir',
        'DecayFastBGToAir',
        'DecayMediumToAir',
        'DecaySlowAGToAir',
        'DecaySlowBGToAir',
        'DecaySWStemSnagToAir',
        'DecaySWBranchSnagToAir',
        'DecayHWStemSnagToAir',
        'DecayHWBranchSnagToAir',
        'DisturbanceMerchToAir',
        'DisturbanceFolToAir',
        'DisturbanceOthToAir',
        'DisturbanceCoarseToAir',
        'DisturbanceFineToAir',
        'DisturbanceDOMCO2Emission',
        'DisturbanceDOMCH4Emission',
        'DisturbanceDOMCOEmission',
        'DisturbanceMerchLitterInput',
        'DisturbanceFolLitterInput',
        'DisturbanceOthLitterInput',
        'DisturbanceCoarseLitterInput',
        'DisturbanceFineLitterInput',
        'DisturbanceVFastAGToAir',
        'DisturbanceVFastBGToAir',
        'DisturbanceFastAGToAir',
        'DisturbanceFastBGToAir',
        'DisturbanceMediumToAir',
        'DisturbanceSlowAGToAir',
        'DisturbanceSlowBGToAir',
        'DisturbanceSWStemSnagToAir',
        'DisturbanceSWBranchSnagToAir',
        'DisturbanceHWStemSnagToAir',
        'DisturbanceHWBranchSnagToAir'
    ]
    ecosystem_decay_emissions_pools = [
        'DecayVFastAGToAir',
        'DecayVFastBGToAir',
        'DecayFastAGToAir',
        'DecayFastBGToAir',
        'DecayMediumToAir',
        'DecaySlowAGToAir',
        'DecaySlowBGToAir',
        'DecaySWStemSnagToAir',
        'DecaySWBranchSnagToAir',
        'DecayHWStemSnagToAir',
        'DecayHWBranchSnagToAir']
    GrossGrowth_pools = [
        'DeltaBiomass_AG',
        'TurnoverMerchLitterInput',
        'TurnoverFolLitterInput',
        'TurnoverOthLitterInput',
        'DeltaBiomass_BG',
        'TurnoverCoarseLitterInput',
        'TurnoverFineLitterInput']

    ecosystem_pools = biomass_pools + dom_pools
    fluxes = ecosystem_decay_emissions_pools
    gross_growth = GrossGrowth_pools
    sum_pools = ['ecosystem', 'biomass', 'DOM']  
    
    pools=sum_pools
    fluxes=['net_emission', 'total_emissions', 'gross_growth']
      
   
    for dtype_key in fm.dtypes:
        dt = fm.dt(dtype_key)
        mask = ('?', '?', '?', '?', dtype_key[4], dtype_key[5])
        for _mask, ytype, curves in fm.yields:
            if _mask != mask: continue # we know there will be a match so this works
            # print('found match for mask', mask)
            pool_data = c_curves_p.loc[' '.join(dtype_key)]
            for yname in pools:
                points = list(zip(pool_data.index.values, pool_data[yname]))
                curve = fm.register_curve(ws3.core.Curve(yname, 
                                                         points=points, 
                                                         type='a', 
                                                         is_volume=False,
                                                         xmax=fm.max_age,
                                                         period_length=fm.period_length))
                curves.append((yname, curve))
                dt.add_ycomp('a', yname, curve)
            flux_data = c_curves_f.loc[' '.join(dtype_key)]
            for yname in fluxes:
                points = list(zip(flux_data.index.values, flux_data[yname]))
                curve = fm.register_curve(ws3.core.Curve(yname, 
                                                         points=points, 
                                                         type='a', 
                                                         is_volume=False,
                                                         xmax=fm.max_age,
                                                         period_length=fm.period_length))
                curves.append((yname, curve))
                dt.add_ycomp('a', yname, curve)
        #mask = '? ? %s ? %' % (dtype_key[2], dtype_key[4])
        #points = c_curves_p


def plugin_c_curves(fm, c_curves_p, c_curves_f):
    # Dictionary to track registered curves for each dtype_key
    # Define Carbon Pools
    biomass_pools = ['SoftwoodMerch','SoftwoodFoliage', 'SoftwoodOther', 'SoftwoodCoarseRoots','SoftwoodFineRoots',                    
                     'HardwoodMerch', 'HardwoodFoliage', 'HardwoodOther', 'HardwoodCoarseRoots', 'HardwoodFineRoots']
    dom_pools = ['AboveGroundVeryFastSoil', 'BelowGroundVeryFastSoil', 'AboveGroundFastSoil', 'BelowGroundFastSoil',
                 'MediumSoil', 'AboveGroundSlowSoil', 'BelowGroundSlowSoil', 'SoftwoodStemSnag', 'SoftwoodBranchSnag',
                 'HardwoodStemSnag', 'HardwoodBranchSnag']
    emissions_pools = ['CO2', 'CH4', 'CO', 'NO2']
    products_pools = ['Products']
    ecosystem_pools = biomass_pools + dom_pools
    all_pools = biomass_pools + dom_pools + emissions_pools + products_pools
    
        # Define Carbon Fluxes
    annual_process_fluxes = [
        'DecayDOMCO2Emission',
        'DeltaBiomass_AG',
        'DeltaBiomass_BG',
        'TurnoverMerchLitterInput',
        'TurnoverFolLitterInput',
        'TurnoverOthLitterInput',
        'TurnoverCoarseLitterInput',
        'TurnoverFineLitterInput',
        'DecayVFastAGToAir',
        'DecayVFastBGToAir',
        'DecayFastAGToAir',
        'DecayFastBGToAir',
        'DecayMediumToAir',
        'DecaySlowAGToAir',
        'DecaySlowBGToAir',
        'DecaySWStemSnagToAir',
        'DecaySWBranchSnagToAir',
        'DecayHWStemSnagToAir',
        'DecayHWBranchSnagToAir'
    ]
    
    npp_fluxes=[
        'DeltaBiomass_AG', 
        'DeltaBiomass_BG'
    ]
    
    decay_emissions_fluxes = [
        'DecayVFastAGToAir',
        'DecayVFastBGToAir',
        'DecayFastAGToAir',
        'DecayFastBGToAir',
        'DecayMediumToAir',
        'DecaySlowAGToAir',
        'DecaySlowBGToAir',
        'DecaySWStemSnagToAir',
        'DecaySWBranchSnagToAir',
        'DecayHWStemSnagToAir',
        'DecayHWBranchSnagToAir'
    ]
    
    disturbance_production_fluxes = [
        'DisturbanceSoftProduction',
        'DisturbanceHardProduction',
        'DisturbanceDOMProduction'   
    ]
    
    disturbance_emissions_fluxes = [
        'DisturbanceMerchToAir',
        'DisturbanceFolToAir',
        'DisturbanceOthToAir',
        'DisturbanceCoarseToAir',
        'DisturbanceFineToAir',
        'DisturbanceVFastAGToAir',
        'DisturbanceVFastBGToAir',
        'DisturbanceFastAGToAir',
        'DisturbanceFastBGToAir',
        'DisturbanceMediumToAir',
        'DisturbanceSlowAGToAir',
        'DisturbanceSlowBGToAir',
        'DisturbanceSWStemSnagToAir',
        'DisturbanceSWBranchSnagToAir',
        'DisturbanceHWStemSnagToAir',
        'DisturbanceHWBranchSnagToAir'   
    ]
    
    all_fluxes = [
        'DisturbanceCO2Production',
        'DisturbanceCH4Production',
        'DisturbanceCOProduction',
        'DisturbanceBioCO2Emission',
        'DisturbanceBioCH4Emission',
        'DisturbanceBioCOEmission',
        'DecayDOMCO2Emission',
        'DisturbanceSoftProduction',
        'DisturbanceHardProduction',
        'DisturbanceDOMProduction',
        'DeltaBiomass_AG',
        'DeltaBiomass_BG',
        'TurnoverMerchLitterInput',
        'TurnoverFolLitterInput',
        'TurnoverOthLitterInput',
        'TurnoverCoarseLitterInput',
        'TurnoverFineLitterInput',
        'DecayVFastAGToAir',
        'DecayVFastBGToAir',
        'DecayFastAGToAir',
        'DecayFastBGToAir',
        'DecayMediumToAir',
        'DecaySlowAGToAir',
        'DecaySlowBGToAir',
        'DecaySWStemSnagToAir',
        'DecaySWBranchSnagToAir',
        'DecayHWStemSnagToAir',
        'DecayHWBranchSnagToAir',
        'DisturbanceMerchToAir',
        'DisturbanceFolToAir',
        'DisturbanceOthToAir',
        'DisturbanceCoarseToAir',
        'DisturbanceFineToAir',
        'DisturbanceDOMCO2Emission',
        'DisturbanceDOMCH4Emission',
        'DisturbanceDOMCOEmission',
        'DisturbanceMerchLitterInput',
        'DisturbanceFolLitterInput',
        'DisturbanceOthLitterInput',
        'DisturbanceCoarseLitterInput',
        'DisturbanceFineLitterInput',
        'DisturbanceVFastAGToAir',
        'DisturbanceVFastBGToAir',
        'DisturbanceFastAGToAir',
        'DisturbanceFastBGToAir',
        'DisturbanceMediumToAir',
        'DisturbanceSlowAGToAir',
        'DisturbanceSlowBGToAir',
        'DisturbanceSWStemSnagToAir',
        'DisturbanceSWBranchSnagToAir',
        'DisturbanceHWStemSnagToAir',
        'DisturbanceHWBranchSnagToAir'
    ]
    
    grossgrowth_ag = [
        "DeltaBiomass_AG",
        "TurnoverMerchLitterInput",
        "TurnoverFolLitterInput",
        "TurnoverOthLitterInput",
    ]
    
    grossgrowth_bg = [
        "DeltaBiomass_BG",
        "TurnoverCoarseLitterInput",
        "TurnoverFineLitterInput",
    ]
    
    product_flux = [
         "DisturbanceSoftProduction",
         "DisturbanceHardProduction",
         "DisturbanceDOMProduction",
    ]
    
    # Define Sum Carbon Pools and Sum Carbon Fluxes
    total_emission = decay_emissions_fluxes + disturbance_emissions_fluxes
    gross_growth = grossgrowth_ag + grossgrowth_bg
    
    sum_pools = ['ecosystem', 'biomass', 'DOM']
    sum_fluxes = ['total_emission', 'gross_growth', 'net_emission']
    
    
    registered_curves = {}

    for dtype_key in fm.dtypes:
        dt = fm.dt(dtype_key)
        mask = ('?', '?', '?', '?', dtype_key[4], dtype_key[5])
        
        for _mask, ytype, curves in fm.yields:
            if _mask != mask: 
                continue  # Only proceed if the mask matches

            # print('found match for mask', mask)

            # Initialize the tracking of registered curves for the dtype_key if not already done
            if dtype_key not in registered_curves:
                registered_curves[dtype_key] = set()

            # Register pool curves
            pool_data = c_curves_p.loc[' '.join(dtype_key)]
            for yname in sum_pools:
                if yname not in registered_curves[dtype_key]:  # Check if curve is already registered
                    points = list(zip(pool_data.index.values, pool_data[yname]))
                    curve = fm.register_curve(ws3.core.Curve(yname, 
                                                             points=points, 
                                                             type='a', 
                                                             is_volume=False,
                                                             xmax=fm.max_age,
                                                             period_length=fm.period_length))
                    curves.append((yname, curve))
                    dt.add_ycomp('a', yname, curve)

                    # Mark the curve as registered
                    registered_curves[dtype_key].add(yname)

            # Register flux curves
            flux_data = c_curves_f.loc[' '.join(dtype_key)]
            for yname in sum_fluxes:
                if yname not in registered_curves[dtype_key]:  # Check if curve is already registered
                    points = list(zip(flux_data.index.values, flux_data[yname]))
                    curve = fm.register_curve(ws3.core.Curve(yname, 
                                                             points=points, 
                                                             type='a', 
                                                             is_volume=False,
                                                             xmax=fm.max_age,
                                                             period_length=fm.period_length))
                    curves.append((yname, curve))
                    dt.add_ycomp('a', yname, curve)

                    # Mark the curve as registered
                    registered_curves[dtype_key].add(yname)






def compile_events(self, softwood_volume_yname, hardwood_volume_yname, n_yield_vals):
    
    def leading_species(dt):
        """
        Determine if softwood or hardwood leading species by comparing softwood and hardwood
        volume at peak MAI age.
        """
        svol_curve, hvol_curve = dt.ycomp(softwood_volume_yname), dt.ycomp(hardwood_volume_yname)
        tvol_curve = svol_curve + hvol_curve
        x_cmai = tvol_curve.mai().ytp().lookup(0)
        return 'softwood' if svol_curve[x_cmai] > hvol_curve[x_cmai] else 'hardwood'

    for dtype_key in self.dtypes:
        dt = self.dt(dtype_key)
        dt.leading_species = leading_species(dt)
    
    theme_cols = [theme['__name__'] for theme in self._themes]
    columns = theme_cols.copy()
    columns += ['species',
                'using_age_class',
                'min_softwood_age',
                'max_softwood_age',
                'min_hardwood_age',
                'max_hardwood_age',
                'MinYearsSinceDist',
                'MaxYearsSinceDist',
                'LastDistTypeID',
                'MinTotBiomassC',
                'MaxTotBiomassC',
                'MinSWMerchBiomassC',
                'MaxSWMerchBiomassC',
                'MinHWMerchBiomassC',
                'MaxHWMerchBiomassC',
                'MinTotalStemSnagC',
                'MaxTotalStemSnagC',	
                'MinSWStemSnagC',
                'MaxSWStemSnagC',
                'MinHWStemSnagC',
                'MaxHWStemSnagC',
                'MinTotalStemSnagMerchC',
                'MaxTotalStemSnagMerchC',
                'MinSWMerchStemSnagC',
                'MaxSWMerchStemSnagC',
                'MinHWMerchStemSnagC',
                'MaxHWMerchStemSnagC',
                'efficiency',
                'sort_type',
                'target_type',
                'target',
                'disturbance_type',
                'disturbance_year']
    data = {c:[] for c in columns}
    for dtype_key, age, area, acode, period, _ in self.compile_schedule():
        #set_trace()
        for i, c in enumerate(theme_cols): data[c].append(dtype_key[i])
        data['species'].append(self.dt(dtype_key).leading_species)
        data['using_age_class'].append('FALSE')
        #############################################################################
        # might need to be more flexible with age range, to avoid OBO bugs and such?
        data['min_softwood_age'].append(-1)
        data['max_softwood_age'].append(-1)
        data['min_hardwood_age'].append(-1)
        data['max_hardwood_age'].append(-1)
        # data['min_softwood_age'].append(age)
        # data['max_softwood_age'].append(age)
        # data['min_hardwood_age'].append(age)
        # data['max_hardwood_age'].append(age)
        #############################################################################
        for c in columns[11:-6]: data[c].append(-1)
        data['efficiency'].append(1)
        data['sort_type'].append(3) # oldest first (see Table 3-3 in the CBM-CFS3 user guide)
        data['target_type'].append('A') # area target
        data['target'].append(area)
        data['disturbance_type'].append(acode)
        data['disturbance_year'].append((period-1)*self.period_length+1)
        # if period == 1:
        #     data['disturbance_year'].append(1)
        # else:
        #     data['disturbance_year'].append((period-1)*self.period_length)
    sit_events = pd.DataFrame(data)         
    return sit_events

def track_system_stock(fm, half_life_solid_wood=30, half_life_paper=2, proportion_solid_wood=1):
    
    product_stock_dict = {}
    solid_wood_stock_list = []
    paper_stock_list = []
    product_stock_list = []
    ecosystem_stock_list = []
    system_stock_list = []

    # Calculate decay rates based on half-lives
    k_solid_wood = math.log(2) / half_life_solid_wood
    k_paper = math.log(2) / half_life_paper

    # Define the allocation distribution
    proportion_paper = 1-proportion_solid_wood

    # Constants
    wood_density = 460 #(Kennedy, 1965)
    carbon_content = 0.5

    for period in fm.periods:
        # Get old product stocks
        last_stocks = next(reversed(product_stock_dict.values()), (0, 0))
        old_product_stock_solid_wood, old_product_stock_paper = last_stocks

        # Calculate new product stocks
        new_product_stock = fm.compile_product(period, 'totvol * 0.85', acode='harvest')* wood_density * carbon_content / 1000 # Convert kg to ton
        new_product_stock_solid_wood = new_product_stock * proportion_solid_wood
        new_product_stock_paper = new_product_stock * proportion_paper 

        # Apply decay to all emissions within the same period they're produced
        sum_product_stock_solid_wood = old_product_stock_solid_wood * (1 - k_solid_wood)**10 + new_product_stock_solid_wood
        sum_product_stock_paper = old_product_stock_paper * (1 - k_paper)**10 + new_product_stock_paper

        # Update product_stock_dict for this period
        product_stock_dict[period] = (sum_product_stock_solid_wood, sum_product_stock_paper)

        # Calculate total system carbon stock
        ecosystem_stock = fm.inventory(period, 'ecosystem')
        sum_product_stock = sum_product_stock_solid_wood + sum_product_stock_paper
        total_system_stock = ecosystem_stock + sum_product_stock

        # Update stock lists for this period
        solid_wood_stock_list.append(sum_product_stock_solid_wood)
        paper_stock_list.append(sum_product_stock_paper)
        product_stock_list.append(sum_product_stock)
        ecosystem_stock_list.append(ecosystem_stock)
        system_stock_list.append(total_system_stock)

    # Prepare data for plotting
    data = {
        'period': fm.periods,
        'solid_wood': solid_wood_stock_list,
        'paper': paper_stock_list,
        'sum_product': product_stock_list,
        'ecosystem': ecosystem_stock_list,
        'system': system_stock_list
    }

    df = pd.DataFrame(data)

    import os
    folder_results_spath = './results'    
    # Check if the folder exists, if not, create it
    if not os.path.exists(folder_results_spath):
        os.makedirs(folder_results_spath)
        # print(f"Folder '{folder_results_spath}' created.")
    # else:
        # print(f"Folder '{folder_results_spath}' already exists.")
    df.to_excel('./results/stock.xlsx', index=False)
    # Plotting
    fig, ax = plt.subplots(1, 5, figsize=(16, 4))  # Adjusted for 5 subplots
    ax[0].bar(df.period, df.solid_wood)
    ax[0].set_title('Solid Wood Product C Stock')
    ax[1].bar(df.period, df.paper)
    ax[1].set_title('Paper Product C Stock')
    ax[2].bar(df.period, df.sum_product)
    ax[2].set_title('Total Product C Stock')
    ax[3].bar(df.period, df.ecosystem)
    ax[3].set_title('Ecosystem C Stock')
    ax[4].bar(df.period, df.system)
    ax[4].set_title('Total System C Stock')

    for a in ax:
        a.set_ylim(None, None)
        a.set_xlabel('Period')
        a.set_ylabel('Stock (tons)')

    plt.tight_layout()
    return fig, ax, df, product_stock_dict

def track_system_emission(fm, half_life_solid_wood=30, half_life_paper=2, proportion_solid_wood=1, displacement_factor=0):
    
    product_stock_dict = {}
    solid_wood_emission_list = []
    paper_emission_list = []
    product_emission_list = []
    ecosystem_emission_list = []
    system_emission_list = []

    # Calculate decay rates based on half-lives
    k_solid_wood = math.log(2) / half_life_solid_wood
    k_paper = math.log(2) / half_life_paper

    # Define the allocation distribution
    proportion_paper = 1-proportion_solid_wood

    # Constants
    wood_density = 460 #(Kennedy, 1965)
    carbon_content = 0.5
    
    for period in fm.periods:
        # Get old product emissions
        last_stocks = next(reversed(product_stock_dict.values()), (0, 0))
        old_product_stock_solid_wood, old_product_stock_paper = last_stocks

        # Calculate new product emissions
        new_product_stock = fm.compile_product(period, 'totvol * 0.85', acode='harvest') * wood_density * carbon_content / 1000 # convert the unit from kg to ton
        new_product_stock_solid_wood = new_product_stock * proportion_solid_wood
        new_product_stock_paper = new_product_stock * proportion_paper

        # Apply decay to all emissions within the same period they're produced
        sum_product_stock_solid_wood = old_product_stock_solid_wood * (1 - k_solid_wood)**10 + new_product_stock_solid_wood
        sum_product_stock_paper = old_product_stock_paper * (1 - k_paper)**10 + new_product_stock_paper

        sum_product_emission_solid_wood = old_product_stock_solid_wood * (1-(1 - k_solid_wood)**10) * 44 / 12 # Convert C to CO2
        sum_product_emission_paper = old_product_stock_paper * (1-(1 - k_paper)**10) * 44 / 12 # Convert C to CO2
        
        # Update product_emission_dict for this period
        product_stock_dict[period] = (sum_product_stock_solid_wood, sum_product_stock_paper)

        # Calculate total system carbon emission
        sum_product_emission = sum_product_emission_solid_wood + sum_product_emission_paper
        ecosystem_emission = (fm.inventory(period-1, 'ecosystem') - fm.inventory(period, 'ecosystem') - new_product_stock) * 44 / 12 if period > 0 else 0
        substitution_effect = new_product_stock_solid_wood*displacement_factor*44/12*-1 # negative emission aviod by displacing high GHG-intensive materials and products with HWPs 
        total_system_emission = ecosystem_emission + sum_product_emission + substitution_effect
        
        # Update stock lists for this period
        solid_wood_emission_list.append(sum_product_emission_solid_wood)
        paper_emission_list.append(sum_product_emission_paper)
        product_emission_list.append(sum_product_emission)
        ecosystem_emission_list.append(ecosystem_emission)
        system_emission_list.append(total_system_emission)

    # Prepare data for plotting
    data = {
        'period': fm.periods,
        'solid_wood': solid_wood_emission_list,
        'paper': paper_emission_list,
        'sum_product': product_emission_list,
        'ecosystem': ecosystem_emission_list,
        'system': system_emission_list
    }

    df = pd.DataFrame(data)
    import os
    folder_results_spath = './results'    
    # Check if the folder exists, if not, create it
    if not os.path.exists(folder_results_spath):
        os.makedirs(folder_results_spath)
        # print(f"Folder '{folder_results_spath}' created.")
    # else:
        # print(f"Folder '{folder_results_spath}' already exists.")
    df.to_excel('./results/emission.xlsx', index=False)

    # Plotting
    fig, ax = plt.subplots(1, 5, figsize=(16, 4))  # Adjusted for 5 subplots
    ax[0].bar(df.period, df.solid_wood)
    ax[0].set_title('Solid Wood Product CO2 Emission')
    ax[1].bar(df.period, df.paper)
    ax[1].set_title('Paper Product CO2 Emission')
    ax[2].bar(df.period, df.sum_product)
    ax[2].set_title('Total Product CO2 Emission')
    ax[3].bar(df.period, df.ecosystem)
    ax[3].set_title('Ecosystem CO2 Emission')
    ax[4].bar(df.period, df.system)
    ax[4].set_title('Total System CO2 Emission')

    for a in ax:
        a.set_ylim(None, None)
        a.set_xlabel('Period')
        a.set_ylabel('Emission (tons)')

    plt.tight_layout()
    return fig, ax, df




################################################
# KPI indicatores 
################################################
def kpi_age(fm, case_study, obj_mode, scenario_name, base_path='.'):
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import pandas as pd
    
    canfi_map_inverse = {
        '1211': 'AC', 
        '1201': 'AT', 
        '304': 'BL', 
        '1303': 'EP', 
        '500': 'FDI', 
        '402': 'HW',
        '403': 'HM',
        '204': 'PLI', 
        '101': 'SB', 
        '104': 'SE', 
        '105': 'SW', 
        '100': 'SX'
    }
    
    Aspen = ['AC', 'ACT', 'AT', 'EP', 'VB', 'MB', 'AT+SX']
    Bal = ['B', 'BA', 'BG', 'BL']
    Cedar = ['CW', 'YC']
    Alder = ['D', 'DR']
    DougFir = ['F', 'FD', 'FDC', 'FDI']
    Hem = ['H', 'HM', 'HW']
    Pine = ['PA', 'PL', 'PLC', 'PW', 'PLI', 'PY']
    Spruce = ['S', 'SS', 'SW', 'SX', 'SE', 'SXW', 'SB']
    
    def find_corresponding_species(number):
        values = canfi_map_inverse.get(str(number))
        if not values:
            return "No corresponding value found."
        
        values = values.split('+')
        for value in values:
            if value in Aspen:
                return 'Aspen'
            elif value in Bal:
                return 'Bal'
            elif value in Cedar:
                return 'Cedar'
            elif value in Alder:
                return 'Alder'
            elif value in DougFir:
                return 'DougFir'
            elif value in Hem:
                return 'Hem'
            elif value in Pine:
                return 'Pine'
            elif value in Spruce:
                return 'Spruce'
        
        return "No matching set found."
    
    old_growth_data = {0: {}, 10: {}}  # For time periods 0 and 10
    
    bin_edges = np.arange(0, 480, 20)
    colors = {
        'Aspen': '#FF0000',
        'Bal': '#FF8C00',
        'Cedar': '#FFD700',
        'Alder': '#00FF00',
        'DougFir': '#00FFFF',
        'Hem': '#1E90FF',
        'Pine': '#9400D3',
        'Spruce': '#FF00FF'
    }  
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, time_period in enumerate([0, 10]):
        cumulative_hist = np.zeros(len(bin_edges) - 1)
        
        # Dictionary to accumulate histograms by species
        species_hist_data = {species: np.zeros(len(bin_edges) - 1) for species in colors.keys()}
        
        for theme3 in fm.theme_basecodes(3):
            data = fm.age_class_distribution(time_period, mask=f'? ? ? {theme3} ? ?')
            x_values = list(data.keys())
            y_values = list(data.values())
            
            hist, _ = np.histogram(x_values, bins=bin_edges, weights=y_values)
            
            species = find_corresponding_species(theme3)
            species_hist_data[species] += hist

            # Calculate old growth area for this species
            old_growth_area = fm.inventory(time_period, 'ogi', mask=f'? ? ? {theme3} ? ?')
            old_growth_data[time_period][species] = old_growth_data[time_period].get(species, 0) + old_growth_area

        # Plot each species only once with the accumulated histogram data
        for species, hist in species_hist_data.items():
            if np.any(hist):  # Only plot species with non-zero histogram data
                axes[idx].bar(bin_edges[:-1], hist, width=20, bottom=cumulative_hist, color=colors[species], edgecolor='black', alpha=0.7, label=f'Species {species}')
                cumulative_hist += hist
        
        axes[idx].set_xlabel('Age')
        axes[idx].set_ylabel('Area (ha)')
        axes[idx].set_title(f'Age Distribution at time period {time_period}')
        axes[idx].legend()
    
    plt.tight_layout()
    
    # Save the plot
    folder_path = os.path.join('./outputs/fig', case_study)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)   
    file_name = f"{case_study}_{obj_mode}_{scenario_name}_age_distribution.pdf"
    file_path = os.path.join(folder_path, file_name)  
    plt.savefig(file_path)
    plt.show()
    plt.close()   
    print(f"Plot saved to {file_path}")
    
    # Convert old growth data to a DataFrame for better display
    old_growth_df = pd.DataFrame(old_growth_data).fillna(0)
    old_growth_df['Difference'] = old_growth_df[10] - old_growth_df[0]
    
    # Print old growth data as a table
    print("\nOld Growth Data (in hectares). \nNegative value indicates loss of old growth and positive value indicates gain of old growth.")
    print(old_growth_df)
    
    # Print conclusion about diversity change based on difference
    if old_growth_df['Difference'].sum() < 0:
        print(f"\nOld growth has **decreased** by {old_growth_df['Difference'].sum():.2f} hectares from time period 0 to time period 10.")
    else:
        print(f"\nOld growth has **increased** by {old_growth_df['Difference'].sum():.2f} hectares from time period 0 to time period 10.")
    
    # Plot clustered column chart for old growth areas by species for each time period
    fig, ax = plt.subplots(figsize=(7, 6))
    old_growth_df[[0, 10]].plot(kind='bar', color=["#FF8C00", "#9400D3"], ax=ax)
    
    ax.set_title("Old Growth Area by Species (Period 0 vs Period 10)")
    ax.set_xlabel("Species")
    ax.set_ylabel("Old Growth Area (ha)")
    ax.legend(["Period 0", "Period 10"])
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the clustered column chart
    clustered_chart_file = f"{case_study}_{obj_mode}_{scenario_name}_old_growth_comparison.pdf"
    clustered_chart_path = os.path.join(folder_path, clustered_chart_file)
    plt.savefig(clustered_chart_path)
    plt.show()
    plt.close()
    
    print(f"Clustered column plot saved to {clustered_chart_path}")
    
    return old_growth_df


def kpi_species(fm, case_study, obj_mode, scenario_name, base_path='.'):
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import math

    # Species codes mapping and color dictionary
    canfi_map_inverse = {
        '1211': 'AC', '1201': 'AT', '304': 'BL', '1303': 'EP', '500': 'FDI',
        '402': 'HW', '403': 'HM', '204': 'PL', '204': 'PLI', '101': 'SB', 
        '104': 'SE', '105': 'SW', '100': 'SX', '1201': 'AT+SX', '100': 'SX+AT'
    }
    
    # Species groups
    species_groups = {
        'Aspen': ['AC', 'ACT', 'AT', 'EP', 'VB', 'MB', 'AT+SX'],
        'Bal': ['B', 'BA', 'BG', 'BL'],
        'Cedar': ['CW', 'YC'],
        'Alder': ['D', 'DR'],
        'DougFir': ['F', 'FD', 'FDC', 'FDI'],
        'Hem': ['H', 'HM', 'HW'],
        'Pine': ['PA', 'PL', 'PLC', 'PW', 'PLI', 'PY'],
        'Spruce': ['S', 'SS', 'SW', 'SX', 'SE', 'SXW', 'SB', 'SX+AT']
    }
    
    # Colors for each species group
    colors = {
        'Aspen': '#FF0000', 'Bal': '#FF8C00', 'Cedar': '#FFD700', 
        'Alder': '#00FF00', 'DougFir': '#00FFFF', 'Hem': '#1E90FF', 
        'Pine': '#9400D3', 'Spruce': '#FF00FF'
    }

    # Helper function to determine species group based on species code
    def find_corresponding_species(number):
        code = canfi_map_inverse.get(str(number))
        if code:
            for species, codes in species_groups.items():
                if code in codes:
                    return species
        return "Unknown Species"

    # Calculate Shannon index and species portions
    def calculate_shannon_index(fm, time_period):
        portions = {}
        total_volume = fm.inventory(time_period, 'totvol')
        
        for theme3 in fm.theme_basecodes(3):
            volume = fm.inventory(time_period, 'totvol', mask=f'? ? ? {theme3} ? ?')
            portions[theme3] = volume / total_volume if total_volume > 0 else 0
        
        shannon_index = -sum(
            portions[theme3] * math.log(portions[theme3]) / math.log(len(fm.theme_basecodes(3)))
            for theme3 in portions if portions[theme3] > 0
        )
        
        # Convert theme3 keys to species names
        named_portions = {find_corresponding_species(theme3): value for theme3, value in portions.items()}
        
        return shannon_index, named_portions

    # Calculate for both time periods
    shannon_0, portion_0 = calculate_shannon_index(fm, time_period=0)
    shannon_10, portion_10 = calculate_shannon_index(fm, time_period=10)
    
    print(f"\nShannon Evenness Index for time period 0: {shannon_0:.4f}")
    print(f"Shannon Evenness Index for time period 10: {shannon_10:.4f}")

    # Calculate change in Shannon index
    shannon_difference = shannon_10 - shannon_0
    if shannon_difference < 0:
        print(f"\nDiversity has **decreased** by {abs(shannon_difference) * 100:.2f}% from time 0 to time 10.")
    else:
        print(f"\nDiversity has **increased** by {abs(shannon_difference) * 100:.2f}% from time 0 to time 10.")

    # Prepare data for pie charts (portions of each species for both time periods)
    labels_0 = list(portion_0.keys())
    sizes_0 = list(portion_0.values())
    labels_10 = list(portion_10.keys())
    sizes_10 = list(portion_10.values())
    
    # Create subplots for pie charts (one row, two columns)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Pie chart for time period 0
    axes[0].pie(
        sizes_0, labels=labels_0, 
        colors=[colors[species] for species in labels_0], autopct='%1.1f%%', startangle=140
    )
    axes[0].set_title("Species Distribution at Time Period 0")
    
    # Pie chart for time period 10
    axes[1].pie(
        sizes_10, labels=labels_10, 
        colors=[colors[species] for species in labels_10], autopct='%1.1f%%', startangle=140
    )
    axes[1].set_title("Species Distribution at Time Period 10")
    
    # Unique species for legend
    unique_species = set(labels_0 + labels_10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[species]) for species in unique_species]
    fig.legend(handles, unique_species, loc="upper right", title="Species Present")
    
    # Save figure
    folder_path = os.path.join('./outputs/fig', case_study)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    file_name = f"{case_study}_{obj_mode}_{scenario_name}_species_pie.pdf"
    file_path = os.path.join(folder_path, file_name)
    plt.savefig(file_path)
    plt.show()
    plt.close()
    
    print(f"Pie Charts for Time Periods 0 and 10 saved to {file_path}")
    return portion_10, shannon_10



def kpi_socioeconomic(fm, util=0.85):
    """
    Calculate the job creation (scocial) and provincial government revenues (economic)
based on the total harvested volume 
    """
    job_creation_2021 = 55715.
    bc_revenue_2021 = 1900000.
    harvest_volume_2021 = 52700000

    socio_coef = job_creation_2021 / harvest_volume_2021
    eco_coef = bc_revenue_2021 / harvest_volume_2021    
    
    socio_indicator = int(sum(fm.compile_product(i, f'totvol * {util}') * socio_coef for i in range(1,11)))
    eco_indicator = int(sum(fm.compile_product(i, f'totvol * {util}') * eco_coef for i in range(1,11)))


    print('The scocial indicator (the number of job creation) is: ', socio_indicator)
    print('The economic indicator (the provincial government revenues) is: ', eco_indicator)       

    return socio_indicator, eco_indicator


################################################################
#Old Growth Inventory 
################################################################
def bootstrap_ogi(fm, tvy_name='totvol', ra1_type='cmai', ra2_type='cyld', rc1=[1., 0.], rc2=[1., 0.], max_y=1.,
                  mask=None, yname='ogi', period_length=10):
    """
    Adds a yield component to each development type expressing "old-growthedness".
    f(x) = 0 on the interval [0, ra1*rc1[0]+rc1[1]].
    f(x) is linearly interpolated on the interval [ra2*rc2[0]+rc2[1], ra1*rc1[0]+rc1[1]]
    f(x) = 1 on the interval [ra2*rc2[0]+rc2[1], inf].
    ra1 defaults to age at which total volume MAI curve culminates.
    ra2 defaults to age at which total volume curve culminates.
    """
    mask = mask if mask else ('?', '?', '?', '?', '?', '?')
    fm.ynames.add(yname)
    for dtk in fm.unmask(mask):
        dt = fm.dtypes[dtk]
        yldca = dt.ycomp(tvy_name).ytp().lookup(0)
        maica = dt.ycomp(tvy_name).mai().ytp().lookup(0)
        ra1 = maica if ra1_type=='cmai' else yldca
        ra2 = maica if ra2_type=='cmai' else yldca
        points = [(0, 0.), 
                  (int(ra1*rc1[0]+rc1[1]), 0.), 
                  (int(ra2*rc2[0]+rc2[1]), max_y),
                  (fm.max_age, max_y)]
        #print(dtk, points)
        c = fm.register_curve(ws3.core.Curve(yname, points=points, type='a', is_volume=False, 
                                             xmax=fm.max_age, period_length=period_length))
        #print(dtk, c.points())
        #assert False
        _mask = (mask[0], '?', dtk[2], dtk[3], dtk[4], dtk[5] )
        fm.yields.append((_mask, 'a', [(yname, c)]))
        dt.add_ycomp('a', yname, c)



################################################################
# Plot results
################################################################
def generate_radar_chart(data, case_study, obj_mode, output_dir="./plots/fig"):
    """
    Generate a radar chart from the provided data, save it as an pdf file in a case-study-specific directory, 
    and include case study and objective mode in the file name.

    Parameters:
        data (dict): A dictionary where the keys are column names and the values are lists of data.
        output_dir (str): The base directory where the files will be saved.
        case_study (str): The name of the case study (used to create the folder and as part of the file name).
        obj_mode (str): The objective mode (used as part of the file name).
    """
    case_study_dir = os.path.join(output_dir, case_study)
    os.makedirs(case_study_dir, exist_ok=True)

    df = pd.DataFrame(data)

    for column in df.columns[1:]:
        max_val = df[column].max()
        min_val = df[column].min()
        df[column] = df[column].apply(lambda x: (x - min_val) / (max_val - min_val))

    categories = list(df.columns[1:])
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Plot each scenario
    for i, row in df.iterrows():
        values = row[1:].tolist()
        values += values[:1]  
        if row["Scenarios"] == "Baseline":
            ax.plot(angles, values, label=row["Scenarios"], color="black", linewidth=2)
            ax.fill(angles, values, color="black", alpha=0.2)  
        else:
            ax.plot(angles, values, label=row["Scenarios"], linewidth=2)
            ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.yaxis.grid(True)
    ax.yaxis.set_tick_params(labelsize=10)
    plt.title(f"{obj_mode}", fontsize=16, pad=30)

    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

    plt.tight_layout()

    file_name = f"{case_study}_{obj_mode}_radar_chart.pdf"
    file_path = os.path.join(case_study_dir, file_name)
    plt.savefig(file_path, format='pdf') 
    plt.show()
    plt.close(fig) 
    print(f"Chart saved at: {file_path}")

def generate_subplots_radar_chart(case_study, obj_modes, data_sets):
    """
    Generate a radar chart with 4 subplots for each obj_mode of the selected case study.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), subplot_kw=dict(polar=True))
    axes = axes.flatten()  # Flatten for easy iteration

    legend_added = False  # Track if legend has been added

    for i, obj_mode in enumerate(obj_modes):
        # Construct the key dynamically
        data_key = f"{case_study}_{obj_mode}".lower()

        ax = axes[i]
        try:
            # Access the dataset from the data_sets dictionary
            data = data_sets[data_key]

            # Convert the data to a DataFrame and normalize values
            df = pd.DataFrame(data)
            for column in df.columns[1:]:
                max_val = df[column].max()
                min_val = df[column].min()
                df[column] = df[column].apply(lambda x: (x - min_val) / (max_val - min_val))

            # Prepare angles for radar chart
            categories = list(df.columns[1:])
            N = len(categories)
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]  # Close the circle

            # Plot each scenario
            for _, row in df.iterrows():
                values = row[1:].tolist()
                values += values[:1]

                # Check if the scenario is "baseline"
                if row['Scenarios'].lower() == "baseline":
                    ax.plot(angles, values, label=row['Scenarios'], color="black", linewidth=2)
                else:
                    ax.plot(angles, values, label=row['Scenarios'])
                    ax.fill(angles, values, alpha=0.1)

            # Add chart features
            ax.set_title(obj_mode, fontsize=14, pad=20)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=10)
            ax.yaxis.grid(True)
            ax.yaxis.set_tick_params(labelsize=8)

            # Add legend to the first subplot only
            if not legend_added:
                ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.2), fontsize=10)
                legend_added = True
        except KeyError:
            print(f"Dataset for {data_key} not found. Skipping...")

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_dir = f'./plots/fig/{case_study}'
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"{case_study}_radar_subplots.pdf"
    file_path = os.path.join(output_dir, file_name)
    plt.savefig(file_path, format='pdf')
    plt.show()
    print(f"Radar chart subplots saved at: {file_path}")

def create_grouped_bar_chart(data, y_label, case_study):
    # Extract scenarios and objective modes
    scenarios = data["Scenarios"]
    modes = list(data.keys())[1:]  # Exclude "Scenarios"

    # Prepare data
    values = [data[mode] for mode in modes]
    x = np.arange(len(scenarios))  # X locations for the groups
    width = 0.8 / len(modes)  # Width of each bar

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, mode in enumerate(modes):
        # Extract the i-th mode's values for all scenarios
        mode_values = [value if value is not None else 0 for value in values[i]]
        ax.bar(x + i * width, mode_values, width, label=mode)

    # Add labels, title, and legend
    ax.set_xticks(x + (len(modes) - 1) * width / 2)
    ax.set_xticklabels(scenarios, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel(y_label, fontsize=12)
    ax.legend(title="Objectives", fontsize=9)

    plt.tight_layout()
    output_dir = f'./plots/fig/{case_study}'
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"{case_study}_{y_label}_indicators_subplots.pdf"
    file_path = os.path.join(output_dir, file_name)
    plt.savefig(file_path, format='pdf')
    plt.show()
    print(f"Radar chart subplots saved at: {file_path}")


