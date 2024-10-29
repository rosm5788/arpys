import numpy as np 
import xarray as xr
import glob

#probably doesn't work, haven't taken any spin data to need to overhaul this
#but prob an easy fix by following spectra_txt_import - Robert
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

#pass only one file to this fn, for maps just use the .zip loader 
def load_spectra_txt(glob_filename):
    textfile = open(glob_filename ,"r")
    text_list = textfile.readlines()
    textfile.close()
    info_dictionary = []
    dim_names = []
    dim_sizes = []
    dim_vals = []
    data_locs = []
    for num,line in enumerate(text_list):
        if "[Info 1]" in line:
            info_row_num = num
            #print("found info", info_row_num)
        elif "Data" in line:
            data_locs.append(num)
            #print("found data", data_locs)

        if "Dimension" in line:
            if "name" in line:
                dim_names.append(line.split("=")[1])
            elif "size" in line:
                dim_sizes.append(int(line.split("=")[1]))
            elif "scale" in line:
                dim_vals.append(np.float64(line.split('=')[1].split()))
             
    dataset = np.zeros(dim_sizes)

    for element in text_list[info_row_num:data_locs[0]]:
        #print("current element",element)
        if element == '\n':
            continue
        note = element[0:-1].split("=")
        if len(note) == 2:
            info_dictionary.append((note[0], note[1]))

    # Data import
    if len(dim_names)>2:
        for dim3_index,start_num in enumerate(data_locs):
            for dim1_index,datarow in enumerate(text_list[start_num+1:start_num+1+dim_sizes[1]]):
                if datarow == '\n':
                    continue
                elif "Data" in datarow:
                    break
                else:
                    dataset[dim1_index,:,dim3_index] = np.float64(datarow.split()[0:-1])
                    #print("adding element ", dim1_index,dim3_index)
    else:
        for dim1_index,datarow in enumerate(text_list[data_locs[0]+1:]):
                if datarow == '\n':
                    continue
                else:
                    dataset[dim1_index,:] = np.float64(datarow.split()[0:-1])
                    #print("adding row ", dim1_index)


    # Seems the data has one more point than the dimensions.. Not sure why the mismatch
    #dataset = dataset[0:-1, 0:-1, :]
    flat_coords = {}
    # For case with only one file
    for i,name in enumerate(dim_names):
        flat_coords[name] = dim_vals[i]
    #print("made coords",flat_coords)
    out = xr.DataArray(
        data= dataset,
        dims= dim_names,
        coords= flat_coords,
        attrs= info_dictionary
    )
    if "Region Iteration[a.u.]\n" in out.dims:
        out = out.sum("Region Iteration[a.u.]\n")
    return out.rename({'Y-Scale [deg]\n':'slit','Kinetic Energy [eV]\n':'energy'}) 
    
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