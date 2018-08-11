import numpy as np
from pyhdf.SD import SD, SDC


class Hdf4File:
    def __init__(self, filename):
        self.filename = filename
        self.file_handle = SD(self.filename, SDC.READ)

class Caliop01kmclay(Hdf4File):
    def __init__(self, filename):
        super().__init__(filename)

    def get_latitudes(self):
        return self.file_handle.select('Latitude')[:]

    def get_longitudes(self):
        return self.file_handle.select('Longitude')[:]

    def get_top_altitude(self, c_i, dn):
        return self.file_handle.select('Layer_Top_Altitude')[c_i - dn : c_i + dn + 1, :4]
    def get_base_altitude(self, c_i, dn):
        return self.file_handle.select('Layer_Base_Altitude')[c_i - dn : c_i + dn + 1, :4]
    def get_top_pressure(self, c_i, dn):
        return self.file_handle.select('Layer_Top_Pressure')[c_i - dn : c_i + dn + 1, :4]
    def get_base_pressure(self, c_i, dn):
        return self.file_handle.select('Layer_Base_Pressure')[c_i - dn : c_i + dn + 1,:4]

class ModisMyd03(Hdf4File):
    def __init__(self, filename):
        super().__init__(filename)

    def get_latitudes(self):
        return self.file_handle.select('Latitude')[:, :]

    def get_longitudes(self):
        return self.file_handle.select('Longitude')[:, :]

class ModisMyd021km(Hdf4File):

    def __init__(self, filename):
        super().__init__(filename)

    def get_input_data(self, c_i, c_j, dn):
        print(c_i, c_j)
        bands = [20, 27, 28, 29, 31, 32, 33]
        band_offsets = [20, 21, 21, 21, 21, 21, 21]
        ds_name = "EV_1KM_Emissive"

        raw_data = self.file_handle.select(ds_name)
        data = raw_data

        res = np.zeros((len(bands), 2 * dn + 1, 2 * dn + 1))

        attributes = raw_data.attributes()
        valid_range = attributes["valid_range"]
        offsets = attributes["radiance_offsets"]
        scales  = attributes["radiance_scales"]
        fill_value = attributes["_FillValue"]

        for i, (b, o) in enumerate(zip(bands, band_offsets)):
            valid_min = valid_range[0]
            valid_max = valid_range[1]
            offset = offsets[b - o]
            scale_factor = scales[b - o]

            res[i, :, :] = data[int(b - o),
                                int(c_i - dn) : int(c_i + dn + 1),
                                int(c_j - dn) : int(c_j + dn + 1)]
            invalid = np.logical_or(res[i, :, :] > valid_max,
                                    res[i, :, :] < valid_min)
            invalid = np.logical_or(invalid, res[i, :, :] == fill_value)
            res[i, invalid] = np.nan

            res[i, :, :] = (res[i, :, :] - offset) * scale_factor
        return res






