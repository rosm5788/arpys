import numpy as np 
import xarray as xr
import io
from zipfile import ZipFile
import glob

def spin_txt_import(txtfile): # She ain't pretty or very smart but she gets the job done for all the single .txt files I have to work with from Ferrum runs
    with open(txtfile) as file:
        text = file.readlines()
    attrs_mode = False
    signal_mode = False
    spin_signals = None
    attrs = {}
    spin_data = []
    for line in text:
        stripped = line.strip()
        if spin_signals is None and stripped.startswith("Signal names"):
            spin_signals = line.strip().split('=')[1].split(' ')
            continue
        if stripped.startswith("[Info 1]"):
            attrs_mode = True
            continue
        if stripped.startswith("[Signal 1]"):
            signal_mode = True
            continue
        if stripped == "":
            attrs_mode = False
            signal_mode = False
            continue
        if attrs_mode:
            if stripped.startswith("["): # Ignores the [Run Mode Information 1] line
                continue
            attr_name, attr_value = stripped.split("=")
            try: # If the attribute is a number, let's make it a number
                attr_value = float(attr_value)
            except ValueError:
                pass 
            attrs[attr_name] = attr_value
        if signal_mode:
            spin_data.append([float(x) for x in stripped.split()])
    spin_data = np.array(spin_data)
    spin_xarray = xr.Dataset({spin_signals[0]: ('energy', spin_data[:,1]), 
                              spin_signals[1]: ('energy', spin_data[:,2])}, 
                              coords={'energy': spin_data[:,0]},attrs=attrs)
    return spin_xarray

def spin_map_zip_import(zipfile): # Only confirmed with maps from Dec 2025, modified ses zip loader
    def read_main_ini_zipped(zipfile, MAIN_INI):
        attrs = {}
        with io.TextIOWrapper(zipfile.open(MAIN_INI), encoding="utf-8") as main_ini:
            for line in main_ini:
                split = line.split("=")
                if len(split) > 1:
                    key = split[0]
                    value = split[1].strip()
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                    attrs[key] = value
        return attrs
    
    input_zip = ZipFile(zipfile)
    _region_ini = 'viewer.ini' # This is the case for all of the spin maps I can find, sorry if that's not the case univerally
    _main_ini = ""
    for name in input_zip.namelist():
        if not name.startswith('viewer') and name.endswith('.ini'):
            _main_ini = name

    #Read REGION_INI
    with io.TextIOWrapper(input_zip.open(_region_ini), encoding="utf-8") as region_ini:
        widthoffset = 0
        spin_names = []
        spin_files = []
        for line in region_ini:
            l = line
            match l:
                case str(x) if x.startswith("width_offset="):
                    widthoffset = float(x.split("=")[1])
                case str(x) if x.startswith("width_delta="):
                    widthdelta = float(x.split("=")[1])
                case str(x) if x.startswith("width="):
                    widthnum = int(x.split("=")[1])
                case str(x) if x.startswith("height_offset="):
                    heightoffset = float(x.split("=")[1])
                case str(x) if x.startswith("height_delta="):
                    heightdelta = float(x.split("=")[1])
                case str(x) if x.startswith("height="):
                    heightnum = int(x.split("=")[1])
                case str(x) if x.startswith("depth_offset="):
                    depthoffset = float(x.split("=")[1])
                case str(x) if x.startswith("depth_delta="):
                    depthdelta = float(x.split("=")[1])
                case str(x) if x.startswith("depth="):
                    depthnum = int(x.split("=")[1])
                case str(x) if x.startswith("width_label="):
                    widthlabel = str(x.split("=")[1].strip())
                case str(x) if x.startswith("height_label="):
                    heightlabel = str(x.split("=")[1].strip())
                case str(x) if x.startswith("depth_label="):
                    depthlabel = str(x.split("=")[1]).strip()
                case str(x) if x.startswith("name="): # I know this isn't the most edge case friendly but in my defense I have no idea what other edge cases there could be
                    if (not str(x).split("=")[1].startswith("SES")) and str(x).split("=")[1].strip() != _main_ini.split(".")[0]:
                        spin_names.append(str(x.split("=")[1]).strip())
                case str(x) if x.startswith("path=") and x.endswith('.bin\n'):
                    spin_files.append(str(x.split("=")[1]).strip())
                case str(x) if x.startswith("ini_path="):
                    _main_ini = str(x.split("=")[1]).strip()

    match str(widthlabel):
        case "Kinetic Energy [eV]":
            energy = np.linspace(widthoffset, widthoffset + widthnum * widthdelta, num=widthnum, endpoint=False)
        case "Thetax [deg]":
            slit = np.linspace(widthoffset, widthoffset + widthnum * widthdelta, num=widthnum, endpoint=False)
        case "Thetay [deg]":
            perp = np.linspace(widthoffset, widthoffset + widthnum * widthdelta, num=widthnum, endpoint=False)

    match heightlabel:
        case "Kinetic Energy [eV]":
            energy = np.linspace(heightoffset, heightoffset + heightnum * heightdelta, num=heightnum, endpoint=False)
        case "Thetax [deg]":
            slit = np.linspace(heightoffset, heightoffset + heightnum * heightdelta, num=heightnum, endpoint=False)
        case "Thetay [deg]":
            perp = np.linspace(heightoffset, heightoffset + heightnum * heightdelta, num=heightnum, endpoint=False)

    match depthlabel:
        case "Kinetic Energy [eV]":
            energy = np.linspace(depthoffset, depthoffset + depthnum * depthdelta, num=depthnum, endpoint=False)
        case "Thetax [deg]":
            slit = np.linspace(depthoffset, depthoffset + depthnum * depthdelta, num=depthnum, endpoint=False)
        case "Thetay [deg]":
            perp = np.linspace(depthoffset, depthoffset + depthnum * depthdelta, num=depthnum, endpoint=False)

    # Read MAIN_INI for attributes and metadata
    attrs = read_main_ini_zipped(input_zip, _main_ini)

    # This then gets all the different spin components and combines them into a dataset (one should be all zeroes based on the Dec 2025 beamtime)
    spin_xarrays = {}
    for spin_name,spin_path in zip(spin_names,spin_files):
        # Reshape the binary file into the correct shape (this may break on other sets of data, watch out)
        with input_zip.open(spin_path, mode='r') as FS_PATH:
            data = FS_PATH.read()
            binaryfile = np.frombuffer(data, dtype=np.float32)
        data = np.reshape(binaryfile,(widthnum, heightnum, depthnum),order='F')
        spin_xarrays[spin_name] = xr.DataArray(data,coords={'slit':slit,'energy':energy,'perp':perp},dims=('energy','slit','perp'))
    return xr.Dataset(spin_xarrays,attrs=attrs)


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
    
if __name__ == "__main__":
    spin_map_test = spin_map_zip_import(r"C:\Users\ajbal\OneDrive - UCB-O365\Dessau Research\ARPES Data\ALS\Ba112\Ba112_010040.zip")
    print(spin_map_test)