import matplotlib.pyplot as plt


class PlotImage:

    def __init__(self):
        self.fig = None

    def __check_isPrime__(self, num):
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                return False
        return True

    def factorial(self, num):
        fac = {}
        for i in range(2, num + 1):
            while self.__check_isPrime__(i) and (num % i == 0):
                if i not in fac.keys():
                    fac[i] = 1
#                     num//=i
                else:
                    fac[i] += 1
#                     num//=i
                num //= i
        return fac

    def get_true_index(self, max_row, row, col):
        return row * max_row + col + row

    def plot(self, image, max_row, max_col, labels=None, name_to_save=None):
        if len(image) > 10:
            new_image = image[0:6]
            if labels is not None:
                new_labels = labels[0:6]
        else:
            new_image = image
            new_labels = labels
#         if self.__check_isPrime__(num_of_image):
#             num_of_image += 1
        self.fig, self.ax = plt.subplots(max_row, max_col, figsize=(10, 5))
        for row in range(len(self.ax)):
            try:
                for col in range(len(self.ax[row])):
                    true_index = self.get_true_index(max_row, row, col)

                    self.ax[row][col].axis("off")
                    self.ax[row][col].imshow(
                        new_image[true_index], cmap=plt.cm.gray)
                    if labels is not None:
                        self.ax[row][col].set_title(
                            f"Image {true_index}: {new_labels[true_index]}")
                    else:
                        self.ax[row][col].set_title(f"Image {true_index}")
            except BaseException:
                true_index = self.get_true_index(max_row, 0, row)
                self.ax[row].axis("off")
                self.ax[row].imshow(new_image[true_index])
                if labels is not None:
                    self.ax[row].set_title(
                        f"Image {true_index}: {new_labels[true_index]}")
                else:
                    self.ax[row].set_title(f"Image {true_index}")
        if name_to_save:
            self.save_plt(name_to_save)
            #         fig, ax = plt.subplot(

    def save_plt(self, name_to_save="temp.jpg"):
        try:
            self.fig.savefig(name_to_save, bbox_inches='tight', dpi=150)

            plt.close(self.fig)
        except BaseException:
            pass
