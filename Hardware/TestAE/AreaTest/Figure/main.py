import PEAreaBreakdown

# =======================================================================
# These data can be verified by the synthesis test in TestAE/AreaTest/PE
# =======================================================================

dc_result = {
    'W4': {
        'A_FP16': {'PE Total (NoAdd)': 41.8, 'Mult': 13.7, 'SNC': 7.1},
        'A_BF16': {'PE Total (NoAdd)': 44.9, 'Mult': 16.8, 'SNC': 7.1},
        'A_FP32': {'PE Total (NoAdd)': 52.9, 'Mult': 16.8, 'SNC': 7.1}
    }, 
    'W8': {
        'A_FP16': {'PE Total (NoAdd)': 54.7, 'Mult': 16.8, 'SNC': 4.3},
        'A_BF16': {'PE Total (NoAdd)': 58.1, 'Mult': 20.2, 'SNC': 4.3},
        'A_FP32': {'PE Total (NoAdd)': 66.2, 'Mult': 20.2, 'SNC': 4.3}
    }
}


# Generate the "pe_area_breakdown.pdf" file accordingly
PEAreaBreakdown.pe_area_breakdown_plot(dc_result)