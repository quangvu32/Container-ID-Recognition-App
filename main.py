import matplotlib.pyplot as plt
from ultralytics import YOLO

from src.preprocessing import text_detection, predict, create_char_imgs, create_horizontal_strip
from src.tesseract import error_check, run_tesseract_on_image

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

def try_spacing_until_valid(img, component, digits_length, max_spacing=10):
    """
    Tăng spacing dần cho đến khi OCR ra kết quả có độ dài mong muốn.
    """
    for spacing in range(max_spacing + 1):
        horizontal_strip = create_horizontal_strip(img, spacing=spacing)
        if horizontal_strip is None:
            continue
        result = run_tesseract_on_image(horizontal_strip, psm=7)
        if len(result) == digits_length[component]:
            return result, horizontal_strip, spacing
    return result, horizontal_strip, spacing  # Fallback to last attempt


def main(img_path):
    model = YOLO('detect/train/weights/best.pt')

    # Detect and predict
    bboxes, classes, names, confs = text_detection(img_path, model)
    annotated_image = predict(img_path, bboxes)

    # Ground-truth expected lengths
    digits_length = {'size': 4, 'number': 7, 'company': 4}

    for p in range(4, 6):  # thử nhiều padding
        char_batch = create_char_imgs(img_path, bboxes, names, classes, padding=p)
        container_dict = {}

        fig, ax = plt.subplots(1, len(char_batch)+1, figsize=(15, 5))
        ax[0].imshow(annotated_image, cmap='gray')
        ax[0].axis('off')
        ax[0].set_title("Annotated Image")

        for i, img in enumerate(char_batch):
            component = names[classes[i]]

            # Find optimal spacing
            result, horizontal_strip, spacing = try_spacing_until_valid(
                img, component, digits_length
            )

            # Error checking
            container_dict[component] = result
            fixed_container_dict = error_check(container_dict)

            # Plot results
            ax[i+1].imshow(horizontal_strip, cmap='gray')
            ax[i+1].axis('off')
            ax[i+1].set_title(f"Result: {fixed_container_dict[component]}")

            print(f'{fixed_container_dict[component]}, '
                  f'Length: {digits_length[component]}, '
                  f'Spacing: {spacing}, Padding: {p}')

        plt.tight_layout()
        plt.show()
    return fixed_container_dict, horizontal_strip, char_batch


if __name__ == '__main__':
    main('images/2.webp')
    #application()
    #main(args)