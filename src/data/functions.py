import pandas as pd

# files = glob(
#     "../../data/raw/MetaMotion/*.csv")
# data_path = "../../data/raw/MetaMotion/"


# def read_data_from_files(files,data_path):

#     acc_df = pd.DataFrame()
#     gyr_df = pd.DataFrame()

#     acc_set = 1
#     gyr_set = 1

#     for f in files:
        
#         # Extract features from filename
#         participants = f.split("-")[0].replace(data_path, "")
#         label = f.split("-")[1]
#         category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

#         df = pd.read_csv(f)

#         df['participants'] = participants
#         df['label'] = label
#         df['category'] = category

#         # Concat all the accelerometer dataset into one acc_df dataframe
#         if "Accelerometer" in f:
#             df["set"] = acc_set
#             acc_set += 1
#             acc_df = pd.concat([acc_df, df])

#         # Concat all the gyroscrope dataset into one acc_df dataframe
#         if "Gyroscope" in f:
#             df["set"] = gyr_set
#             gyr_set += 1
#             gyr_df = pd.concat([gyr_df, df])
    
#     # Working with datetimes & setting the index as epoch (ms)       
#     acc_df.index = pd.to_datetime(acc_df['epoch (ms)'], unit="ms")
#     gyr_df.index = pd.to_datetime(gyr_df['epoch (ms)'], unit="ms")

#     del acc_df["epoch (ms)"]
#     del acc_df["time (01:00)"]
#     del acc_df["elapsed (s)"]

#     del gyr_df["epoch (ms)"]
#     del gyr_df["time (01:00)"]
#     del gyr_df["elapsed (s)"]

#     return acc_df, gyr_df


# acc_df, gyr_df = read_data_from_files(files,data_path)



# ==========================================================

# Function for contouring and cropping
def contour_and_crop(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image

# Function to resize and normalize
def resize_and_normalize(image, target_size=(128, 128)):
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    normalized = resized / 255.0  # Normalizing to [0, 1]
    return normalized

# Function to preprocess a dataset of images
def preprocess_images(image_dir, target_size=(128, 128)):
    images = []
    labels = []
    for label_folder in os.listdir(image_dir):  # Assuming folder names are class labels
        folder_path = os.path.join(image_dir, label_folder)
        if not os.path.isdir(folder_path):
            continue
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            image = cv2.imread(file_path)
            if image is not None:
                cropped = contour_and_crop(image)
                preprocessed = resize_and_normalize(cropped, target_size)
                images.append(preprocessed)
                labels.append(label_folder)  # Class label
    return np.array(images), np.array(labels)