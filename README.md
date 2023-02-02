# face-aligned-timelapse
With a bunch of face pictures, add date label, align face shape and generates a timelapse video!

## Setup

1. Install the requirements using the `requirements.txt`

```sh
pip install -r requirements.txt
```

## Running

1. Add a folder `/input` with all the pics with the Telegram styled name, such as: `"photo_1@20-03-2021_00-18-52"`.
2. Run the `main.py`

## Process

The code will:

1. Create a folder `/labled` with the pics labled with the day since the first pic and the formatted date.
2. Proccess each pic to align them in the same head position.
3. Renderize the `timelapse.mp4` and save it in the root directory.

:)
