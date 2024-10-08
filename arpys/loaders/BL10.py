import numpy as np 
import xarray as xr
import glob

def spin_txt_import(glob_filenames):

    # Check if input is list or str

    if isinstance(glob_filenames, list):
        n_files = len(glob_filenames)
        textfile = open(glob_filenames[0] ,"r")
        text_list = textfile.readlines()
        textfile.close()
    elif isinstance(glob_filenames, str):
        n_files = 1
        textfile = open(glob_filenames ,"r")
        text_list = textfile.readlines()
        textfile.close()
    else:
        raise NameError('Input either list or str')

    # Hard coded row numbers
    info_row_num = 17
    data_row_num = 52
    ThetaX_row_num = 43
    ThetaY_row_num = 44

    # Dimension 1
    sp_st = text_list[6][0:-1].split('=')
    dim1_name = sp_st[1]
    sp_st = text_list[7][0:-1].split('=')
    dim1_size = int(sp_st[1])
    sp_st = text_list[8][0:-1].split('=')
    dim1 = np.fromstring(sp_st[1], sep = ' ')
    # Dimension 2
    dim2_name = 'ThetaY'
    dim2_size = n_files
    dim2 = np.zeros(n_files)

    # Initialize empty data array
    dataset = np.zeros((dim1_size, dim2_size))

    # Get metadata & make info dictionary 
    info_dictionary = []
    for pt, element in enumerate(text_list[info_row_num:]):
        if element == '\n':
            break
        note = element[0:-1].split("=")
        if len(note) == 2:
            info_dictionary.append((note[0], note[1]))
    info_dictionary = dict(info_dictionary)


    # Import data & ThetaY values in every file
    if isinstance(glob_filenames, list):
        for index, element in enumerate(glob_filenames):
            textfile = open(element ,"r")
            text_list = textfile.readlines()
            textfile.close()

            # Data import
            for data_index, datarow in enumerate(text_list[data_row_num:]):
                if datarow == '\n':
                    break
                temp = np.fromstring(datarow, sep = ' ')
                dataset[data_index, index] = temp[3]
            
            # ThetaY import
            sp_st = text_list[ThetaY_row_num][0:-1].split('=')
            dim2[index] = float(sp_st[1])
    else:
        textfile = open(glob_filenames ,"r")
        text_list = textfile.readlines()
        textfile.close()

        # Data import
        for data_index, datarow in enumerate(text_list[data_row_num:]):
            if datarow == '\n':
                break
            temp = np.fromstring(datarow, sep = ' ')
            dataset[data_index, 0] = temp[3]
            
        # ThetaY import
        sp_st = text_list[ThetaY_row_num][0:-1].split('=')
        dim2[0] = float(sp_st[1])

    coord_dictionary = dict([(dim1_name, dim1), (dim2_name, dim2)])
    flat_coords = dict([(dim1_name, dim1)])
    
    if n_files != 1:
        return xr.DataArray(
            data= dataset,
            dims= [dim1_name, dim2_name],
            coords= coord_dictionary,
            attrs= info_dictionary
        )
    else:
        return xr.DataArray(
            data= dataset[:,0],
            dims= [dim1_name],
            coords= flat_coords,
            attrs= info_dictionary
        )


def spectra_txt_import(glob_filenames):

    # Check if input is list or str

    if isinstance(glob_filenames, list):
        n_files = len(glob_filenames)
        textfile = open(glob_filenames[0] ,"r")
        text_list = textfile.readlines()
        textfile.close()
    elif isinstance(glob_filenames, str):
        n_files = 1
        textfile = open(glob_filenames ,"r")
        text_list = textfile.readlines()
        textfile.close()
    else:
        raise NameError('Input either list or str')

    # Hard coded row numbers
    info_row_num = 17
    data_row_num = 54
    ThetaX_row_num = 48
    ThetaY_row_num = 49

    # Dimension 1
    sp_st = text_list[6][0:-1].split('=')
    dim1_name = sp_st[1]
    sp_st = text_list[7][0:-1].split('=')
    dim1_size = int(sp_st[1]) + 1
    sp_st = text_list[8][0:-1].split('=')
    dim1 = np.fromstring(sp_st[1], sep = ' ')
    # Dimension 2
    sp_st = text_list[9][0:-1].split('=')
    dim2_name = sp_st[1]
    sp_st = text_list[10][0:-1].split('=')
    dim2_size = int(sp_st[1]) + 1
    sp_st = text_list[11][0:-1].split('=')
    dim2 = np.fromstring(sp_st[1], sep = ' ')
    # Dimension 3
    dim3_name = 'ThetaX'
    dim3_size = n_files
    dim3 = np.zeros(n_files)

    # Initialize empty data array
    dataset = np.zeros((dim1_size, dim2_size, n_files))

    # Get metadata & make info dictionary 
    info_dictionary = []
    for pt, element in enumerate(text_list[info_row_num:]):
        if element == '\n':
            break
        note = element[0:-1].split("=")
        if len(note) == 2:
            info_dictionary.append((note[0], note[1]))
    info_dictionary = dict(info_dictionary)

    # Import data & ThetaX values in every file
    if isinstance(glob_filenames, list):
        for index, element in enumerate(glob_filenames):
            textfile = open(element ,"r")
            text_list = textfile.readlines()
            textfile.close()

            # Data import
            for data_index, datarow in enumerate(text_list[data_row_num:]):
                if datarow == '\n':
                    break
                dataset[data_index, :, index] = np.fromstring(datarow, sep = ' ')
            
            # ThetaX import
            sp_st = text_list[ThetaX_row_num][0:-1].split('=')
            dim3[index] = int(sp_st[1])
    else:
        textfile = open(glob_filenames ,"r")
        text_list = textfile.readlines()
        textfile.close()

        # Data import
        for data_index, datarow in enumerate(text_list[data_row_num:]):
            if datarow == '\n':
                break
            dataset[data_index, :, 0] = np.fromstring(datarow, sep = ' ')
        
        # ThetaX import
        sp_st = text_list[ThetaX_row_num][0:-1].split('=')
        dim3[0] = int(sp_st[1])
    

    # Seems the data has one more point than the dimensions.. Not sure why the mismatch
    dataset = dataset[0:-1, 0:-1, :]

    coord_dictionary = dict([(dim1_name, dim1), (dim2_name, dim2), (dim3_name, dim3)])

    # For case with only one file
    flat_coords = dict([(dim1_name, dim1), (dim2_name, dim2)])

    if n_files != 1:
        out = xr.DataArray(
            data= dataset,
            dims= [dim1_name, dim2_name, dim3_name],
            coords= coord_dictionary,
            attrs= info_dictionary
        )
        return out.rename({'Y-scale [deg]':'slit'})
    else:
        out = xr.DataArray(
            data= dataset[:,:,0],
            dims= [dim1_name, dim2_name],
            coords= flat_coords,
            attrs= info_dictionary
        )
        return out.rename({'Y-Scale [deg]':'slit','Kinetic Energy [eV]':'energy'}) 
    
    
#didn't work for Feb 2023 data, updated above w/ new hard coded row numbers
def spectra_txt_import_old(glob_filenames):

    # Check if input is list or str

    if isinstance(glob_filenames, list):
        n_files = len(glob_filenames)
        textfile = open(glob_filenames[0] ,"r")
        text_list = textfile.readlines()
        textfile.close()
    elif isinstance(glob_filenames, str):
        n_files = 1
        textfile = open(glob_filenames ,"r")
        text_list = textfile.readlines()
        textfile.close()
    else:
        raise NameError('Input either list or str')

    # Hard coded row numbers
    info_row_num = 14
    data_row_num = 51
    ThetaX_row_num = 44
    ThetaY_row_num = 45

    # Dimension 1
    sp_st = text_list[6][0:-1].split('=')
    dim1_name = sp_st[1]
    sp_st = text_list[7][0:-1].split('=')
    dim1_size = int(sp_st[1]) + 1
    sp_st = text_list[8][0:-1].split('=')
    dim1 = np.fromstring(sp_st[1], sep = ' ')
    # Dimension 2
    sp_st = text_list[9][0:-1].split('=')
    dim2_name = sp_st[1]
    sp_st = text_list[10][0:-1].split('=')
    dim2_size = int(sp_st[1]) + 1
    sp_st = text_list[11][0:-1].split('=')
    dim2 = np.fromstring(sp_st[1], sep = ' ')
    # Dimension 3
    dim3_name = 'ThetaX'
    dim3_size = n_files
    dim3 = np.zeros(n_files)

    # Initialize empty data array
    dataset = np.zeros((dim1_size, dim2_size, n_files))

    # Get metadata & make info dictionary 
    info_dictionary = []
    for pt, element in enumerate(text_list[info_row_num:]):
        if element == '\n':
            break
        note = element[0:-1].split("=")
        if len(note) == 2:
            info_dictionary.append((note[0], note[1]))
    info_dictionary = dict(info_dictionary)

    # Import data & ThetaX values in every file
    if isinstance(glob_filenames, list):
        for index, element in enumerate(glob_filenames):
            textfile = open(element ,"r")
            text_list = textfile.readlines()
            textfile.close()

            # Data import
            for data_index, datarow in enumerate(text_list[data_row_num:]):
                if datarow == '\n':
                    break
                dataset[data_index, :, index] = np.fromstring(datarow, sep = ' ')
            
            # ThetaX import
            sp_st = text_list[ThetaX_row_num][0:-1].split('=')
            dim3[index] = int(sp_st[1])
    else:
        textfile = open(glob_filenames ,"r")
        text_list = textfile.readlines()
        textfile.close()

        # Data import
        for data_index, datarow in enumerate(text_list[data_row_num:]):
            if datarow == '\n':
                break
            dataset[data_index, :, 0] = np.fromstring(datarow, sep = ' ')
        
        # ThetaX import
        sp_st = text_list[ThetaX_row_num][0:-1].split('=')
        dim3[0] = int(sp_st[1])
    

    # Seems the data has one more point than the dimensions.. Not sure why the mismatch
    dataset = dataset[0:-1, 0:-1, :]

    coord_dictionary = dict([(dim1_name, dim1), (dim2_name, dim2), (dim3_name, dim3)])

    # For case with only one file
    flat_coords = dict([(dim1_name, dim1), (dim2_name, dim2)])

    if n_files != 1:
        return xr.DataArray(
            data= dataset,
            dims= [dim1_name, dim2_name, dim3_name],
            coords= coord_dictionary,
            attrs= info_dictionary
        )
    else:
        return xr.DataArray(
            data= dataset[:,:,0],
            dims= [dim1_name, dim2_name],
            coords= flat_coords,
            attrs= info_dictionary
        )