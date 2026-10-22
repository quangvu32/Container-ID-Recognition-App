import matplotlib.pyplot as plt
from ultralytics import YOLO
#from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import text_detection, predict, create_char_imgs, create_horizontal_strip
from src.error_check import error_check
from src.ocr_crnn import run_text_recognition_crnn

# def argument_parser():
#     '''Handle command line arguments'''
#     ap = argparse.ArgumentParser()
#     ap.add_argument('-w', '--weights', type=str, default='detect/train/weights/best.pt',
#                     help='Path to the model weights')
#     ap.add_argument('-i', '--image', type=str, default='images/1.webp',
#                     help='Path to the input image')
#     ap.add_argument('-s', '--spacing', type=int, default=5,
#                     help='Spacing between characters in the horizontal strip')
#     args = ap.parse_args()
#     return args

def main(img_path):
    model = YOLO('../best.pt')
    bboxes, classes, names, confs = text_detection(img_path, model)
    annotated_image= predict(img_path, bboxes)
    char_batch = create_char_imgs(img_path, bboxes)

    # Ground-truth expected lengths
    digits_length = {'size': 4, 'number': 7, 'company': 4}
    container_dict = {}

    fig, ax = plt.subplots(1, len(char_batch)+1, figsize=(15, 5))
    ax[0].imshow(annotated_image, cmap='gray')
    ax[0].axis('off')
    ax[0].set_title("Annotated Image")

    for i, img in enumerate(char_batch):
        component = names[classes[i]]

        horizontal_strip = create_horizontal_strip(img)
        if horizontal_strip is None: 
            continue

        result = run_text_recognition_crnn(horizontal_strip)

        # Error checking
        container_dict[component] = result
        fixed_container_dict = error_check(container_dict)

        # Plot results
        ax[i+1].imshow(horizontal_strip, cmap='gray')
        ax[i+1].axis('off')
        ax[i+1].set_title(f"Result: {fixed_container_dict[component]}")

        print(f'{component}: '.title(), f'{fixed_container_dict[component]}, ' f'Length: {digits_length[component]}')

    plt.tight_layout()
    plt.show()
    return fixed_container_dict, horizontal_strip, char_batch

if __name__ == '__main__':
    main('images/1.webp')
    #application()
    #main(args)