import numpy as np 
import xarray as xr

def spin_txt_import(filename):
    textfile = open(filename ,"r")
    text_list = textfile.readlines()
    textfile.close()

    # How many edc's in this file
    temp = text_list[1]
    temp = temp[0:-1].split("=")
    n_scan = int(temp[1])


    # Find the beginning rows of region sections
    region_start_rows = np.zeros(n_scan)
    for index, dummy in enumerate(region_start_rows):
        string_target = '[Region ' + str(index + 1) + ']'
        for ct, element in enumerate(text_list):
            if element.find(string_target) != -1:
                region_start_rows[index] = ct + 1
                


    # Make region dictionary
    region_dictionary = []
    for index, row in enumerate(region_start_rows):
        for element in text_list[int(row):]:
            if element == '\n':
                break
            note = element[0:-1].split("=")
            if len(note) == 2:
                region_dictionary.append((note[0]+str(index + 1), note[1]))
    region_dictionary = dict(region_dictionary)

    num_steps = np.zeros(n_scan)
    for index, dummy in enumerate(num_steps):
        num_steps[index] = int(region_dictionary['Signal steps' + str(index+1)])


    # Find the beginning rows of info sections
    info_start_rows = np.zeros(n_scan)
    for index, dummy in enumerate(info_start_rows):
        string_target = '[Info ' + str(index + 1) + ']'
        for ct, element in enumerate(text_list):
            if element.find(string_target) != -1:
                info_start_rows[index] = ct + 1


    # Make info dictionary
    info_dictionary = []
    for index, row in enumerate(info_start_rows):
        for pt, element in enumerate(text_list[int(info_start_rows[index]):]):
            if element == '\n':
                break
            note = element[0:-1].split("=")
            if len(note) == 2:
                info_dictionary.append((note[0]+str(index + 1), note[1]))
    info_dictionary = dict(info_dictionary)


    # Find the beginning rows of data sets
    data_start_rows = np.zeros(n_scan)

    for index, dummy in enumerate(data_start_rows):
        string_target = '[Signal ' + str(index + 1) + ']'
        for ct, element in enumerate(text_list):
            if element.find(string_target) != -1:
                data_start_rows[index] = ct + 1


    # Make data array
    dataset = np.zeros((n_scan,int(num_steps[0])))
    edc = np.zeros((int(num_steps[0])))

    for index, row in enumerate(data_start_rows):
        for pt, element in enumerate(text_list[int(row):]):
            if element == '\n':
                break
            temp_row = np.fromstring(element, sep = ' ')
            ydiff = temp_row[3]
            eV = temp_row[0]
            edc[pt] = ydiff
        dataset[index, :] = edc


    # Make Energy range and Angle range for xarray coordinates
    e_range_str = region_dictionary['Dimension 1 scale1']
    e_range = np.fromstring(e_range_str, sep = ' ')
    a_range = np.zeros(n_scan)
    for index, dummy in enumerate(a_range):
        a_range[index] = info_dictionary['ThetaY'+str(index+1)]


    coord_dictionary = dict([(region_dictionary['Dimension 1 name1'], e_range), ('ThetaY', a_range)])

    return xr.DataArray(
        data= dataset,
        dims= ['ThetaY', region_dictionary['Dimension 1 name1']],
        coords= coord_dictionary,
        attrs= info_dictionary
    )
    