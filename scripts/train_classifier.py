training_data = sys.argv[1]
output_path = sys.argv[2]

training_data = Dataset(training_data)
x = training_data.variables["input"][:]
y = training_data.variables["cloud_class"][:]

#
# Remove nans
#

inds = np.where(np.all(np.logical_not(np.isnan(x)), axis = (1, 2, 3)))[0]
x = x[inds, :, :, :]
y = y[inds]


model = CloudNetDetection(20, layers = 4, dense_width = 128)
model.fit(x, y)
model.save(os.path.join(output_path, "cloud_detector_" + str(id(model))))
