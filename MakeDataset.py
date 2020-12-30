
import tensorflow as tf 


# Use the tf function to accelerate the image reading speed

@tf.function
def Read_image(image_paths):

    img = tf.io.read_file(image_paths)
    img = tf.image.decode_jpeg(img , channels = 3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    return img


# Use the tensorflow built func, dataset, to generate a iterator

def MakeDataset(image_paths, label_paths, shuffle_num, batch=8, num_core=tf.data.experimental.AUTOTUNE):

    img_dataset = tf.data.Dataset.from_tensor_slices(image_paths)

    mask_dataset = tf.data.Dataset.from_tensor_slices(label_paths)

    img_dataset = img_dataset.map(map_func=Read_image, num_parallel_calls=num_core)

    mask_dataset = mask_dataset.map(map_func=Read_image, num_parallel_calls=num_core)

    Dataset = tf.data.Dataset.zip((img_dataset, mask_dataset))

    Dataset = Dataset.shuffle(shuffle_num).batch(batch).prefetch(num_core)

    return Dataset