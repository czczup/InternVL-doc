# Chat Data Format

## Introduction

In this document, we will detail the organization format of our conversation data. Currently, we use a JSON file to manage the meta information of all datasets. The format is as follows:

```
{
  "your-custom-dataset-1": {
    "root": "path/to/the/image/",
    "annotation": "path/to/the/jsonl/annotation",
    "data_augment": false,
    "repeat_time": 1,
    "length": "number of your data"
  },
  ...
}
```

Here, `root` is the root directory of the dataset, `annotation` is the path to the annotation file, `data_augment` indicates whether data augmentation is needed, `repeat_time` is the number of times the dataset is repeated, and `length` is the number of samples in the dataset.

For example, a file for the ShareGPT4V dataset looks like this:

```
{
  "sharegpt4v_instruct_gpt4-vision_cap100k": {
    "root": "playground/data/",
    "annotation": "playground/opensource/sharegpt4v_instruct_gpt4-vision_cap100k.jsonl",
    "data_augment": false,
    "repeat_time": 1,
    "length": 102025
  },
  ...
}
```

You can add multiple datasets in this JSON file, similar to [this file](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/data/internvl_1_2_finetune.json).

We currently support the following four types of datasets: `pure text data`, `single image data`, `multi-image (interleaved) data`, and `video data`. We do not require all entries in a JSONL file to be of the same type, meaning your JSONL file can contain different types of data.

## Pure Text Data

For pure text data, we use a JSONL file to store the data. Each entry is a dictionary organized in the following format. Note that entries for pure text data should not contain an `image` field.

```json
{
  "id": 0,
  "conversations": [
    {"from": "human", "value": "user input"},
    {"from": "gpt", "text": "assistant output"},
    {"from": "human", "value": "user input"},
    {"from": "gpt", "text": "assistant output"}
  ]
}
```

Here, `id` is the unique identifier for the data, and `conversations` is a list containing multiple conversations. Each conversation is a dictionary with two key-value pairs: `from` and `value`. `from` indicates the speaker's identity, and `value` indicates the speaker's input.

This data format is consistent with LLaVA 1.5, with the difference being that LLaVA 1.5 uses JSON, while we use JSONL.

An example of pure text data:

```json
{"id": 0, "conversations": [{"from": "human", "value": "Find the length of the hypotenuse given two sides\n\"side_length1 = 5, side_length2 = 6\""}, {"from": "gpt", "value": "To find the length of the hypotenuse given the two sides of a right triangle, you can use the Pythagorean theorem. The theorem states that the square of the length of the hypotenuse is equal to the sum of the squares of the lengths of the two other sides.\n \nIn this case, the length of one side is 5, and the length of the other side is 6. Plugging these values into the formula, we get:\n\nhypotenuse2 = 5^2 + 6^2\n                 = 25 + 36\n                 = 61\n\nTo get the length of the hypotenuse, take the square root of 61:\n\nhypotenuse = √61\n                 = 7.81 (rounded to two decimal places)\n\nTherefore, the length of the hypotenuse is approximately 7.81 units."}]}
```

## Single Image Data

For single image data, we use a JSONL file to store the data. Each entry is a dictionary organized in the following format. Note that each entry for single image data must contain an `image` field, which is a string.

The path in the `image` field is relative to the `root` field. Concatenating the `root` field and the `image` field gives the complete path to the image. It is recommended to include `width` and `height` information for each data sample for future use.

```json
{
  "id": 0,
  "image": "path/to/image.jpg",
  "width": 111,
  "height": 222,
  "conversations": [
    {"from": "human", "value": "<image>\nuser input"},
    {"from": "gpt", "text": "assistant output"},
    {"from": "human", "value": "user input"},
    {"from": "gpt", "text": "assistant output"}
  ]
}
```

Here, `<image>` indicates the position where the image is inserted, and the number of `<image>` placeholders should match the number of images. In single image data, the `<image>` placeholder should appear only once across all conversations.

An example of single image data:

```json
{"id": 0, "image": "images/00000000.jpg", "conversations": [{"from": "human", "value": "<image>\nCan you extract any readable text from the image?"}, {"from": "gpt", "value": "Dares Wins Vol. 5 Tommy's Heroes Vol. 6: For Tomorrow Vol. 7: Closing Time miniseries. Clark Kent is being interviewed about Superman's connection to notorious killer Tommy Monaghan. Taking the conversation..."}], "width": 897, "height": 1152}
```

## Multi-Image Data

For multi-image data, we use a JSONL file to store the data. Each entry is a dictionary organized in the following format. Note that each entry for multi-image data must contain an `image` field, which is a list of strings.

Each element in the list is a path relative to the `root` field. Concatenating the `root` field and each element gives the complete path to the images. It is recommended to include `width_list` and `height_list` information for each data sample for future use.

```json
{
  "id": 0,
  "image": ["path/to/image1.jpg", "path/to/image2.jpg", "path/to/image3.jpg"],
  "width_list": [111, 222, 333],
  "height_list": [111, 222, 333],
  "conversations": [
    {"from": "human", "value": "<image>\nuser input <image>\nuser input"},
    {"from": "gpt", "text": "assistant output"},
    {"from": "human", "value": "<image>\nuser input"},
    {"from": "gpt", "text": "assistant output"}
  ]
}
```

Here, `<image>` indicates the position where the images are inserted, and the number of `<image>` placeholders should match the number of images.
In this example, the `image` field list contains three elements, so the `<image>` placeholder also needs to appear three times.

An example of multi-image data:

```json
{"id": 0, "image": ["cimages/multimages/16/5pc.png", "cimages/multimages/16/5pd.png", "cimages/multimages/16/1602207874_p5b.png", "cimages/multimages/16/5pe.png", "cimages/multimages/16/1473016381_p5a.png"], "height_list": [23, 22, 23, 41, 52], "width_list": [240, 240, 240, 240, 240], "conversations": [{"from": "human", "value": "Let F = {2, 5, 7, 9}\n\nLet G = {1, 4, 6, 8}\n\nWhich of the following is true?\nA. \n<image>\n\nB. /\n<image>\n\nC. /\n<image>\n\nD. /\n<image>\n\nE. /\n<image>\n\nAnswer with the option's letter from the given choices directly."}, {"from": "gpt", "value": "A"}]}
```

## Video Data

For video data, we use a JSONL file to store the data. Each entry is a dictionary organized in the following format. Note that each entry for video data must contain a `video` field, which is a string.

The path in the `video` field is relative to the `root` field. Concatenating the `root` field and the `video` field gives the complete path to the video.

```json
{
  "id": 0,
  "video": "path/to/video.mp4",
  "conversations": [
    {"from": "human", "value": "<video>\nuser input"},
    {"from": "gpt", "text": "assistant output"},
    {"from": "human", "value": "user input"},
    {"from": "gpt", "text": "assistant output"}
  ]
}
```

Here, `<video>` indicates the position where the video is inserted, and the number of `<video>` placeholders should match the number of videos. In video data, the `<video>` placeholder should appear only once across all conversations.

An example of video data:

```json
{"id": 0, "video": "000001_000050/1066682446.mp4", "conversations": [{"from": "human", "value": "<video>\nWrite an exhaustive depiction of the given video, capturing its essence and key moments."}, {"from": "gpt", "value": "First, the video shows a black clock with black numbers and a white background on the side of a building. Next, the numbers show the time. Then, the clock is shown in black and white, but the numbers still show the time. The clock now displays the time as 830. Finally, a black clock with numbers is on the side of a tower, displaying the date as January 28. Overall, the video is a 3D animation of a flip clock calendar for the date of January 28, showing the different stages of the clock's display as it flips through the time and date."}]}
```

We currently do not support datasets that interleave video, image, and text or datasets with multiple videos. Support for such datasets may be added in the future if there is a need.

<br>
<br>