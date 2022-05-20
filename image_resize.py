import os
import gc
import uuid
import fitz
import shutil
import logging

import numpy as np
from pikepdf import Pdf
from pdf2image import convert_from_path
from PyPDF2 import PdfFileWriter, PdfFileReader


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResizePdfFiles(object):

    def __init__(self, input_folder_path, sizes):
        self.change_working_directory()
        self.temp_pdf_output_dir = "./tmp_splitted_pdf"
        self.temp_pdf_resized_output_dir = "./tmp_resized_splitted_pdf"
        self.check_new_image_size = "./tmp_check_size/"
        self.final_pdf_output_dir = "./final_pdf"
        self.final_image_output_dir = "./final_image"
        self.doc_splitted_pdf_folder = ""
        self.doc_resized_splitted_pdf_folder = ""
        self.doc_final_pdf_folder = ""
        self.doc_final_image_folder = ""
        self.input_folder_path = input_folder_path
        self.sizes = sizes
        self.original_file_size = None

    def resize(self):
        files_path = self.get_files_list()
        for file_path in files_path:
            logger.info("Starting resizing process for file: {}".format(os.path.basename(file_path.split(".")[0])))
            if os.path.basename(file_path.split(".")[1]).lower() == 'pdf':
                self.resize_pdf_document(file_path)
            else:
                logger.info("File is not a PDF")
            logger.info("Deleting tmp folders started")
            self.delete_folders()
            logger.info("Deleting tmp folders finished")
            logger.info("Deleting memory values started")
            gc.collect()
            logger.info("Deleting memory values finished")
            logger.info("")
            logger.info("")
            logger.info("")

    def resize_pdf_document(self, original_file_path):
        try:
            logger.info("Resizing pdf file started")
            logger.info("Checking PDF pages size started")
            needs_resizing = self.check_pdf_pages_size(original_file_path)
            logger.info("Checking PDF pages size finished")
            if needs_resizing:
                logger.info("PDF file needs to be resized")
                logger.info("Splitting pdf file started")
                files_path = self.split_pdf(original_file_path)
                logger.info("Splitting pdf file finished. Number of pages {}".format(len(files_path)))
                if files_path:
                    logger.info("Checking image size and resizing started")
                    pdf_paths = self.check_image_size_resize(files_path, original_file_path)
                    logger.info("Checking image size and resizing finished")
                    if pdf_paths:
                        logger.info("Grouping resized pdfs into single file started")
                        resized_pdf_path = self.group_pdfs(pdf_paths, original_file_path)
                        logger.info("Grouping resized pdfs into single file finished")
                        logger.info("Resizing pdf file finished")
                        return resized_pdf_path
                    return ""
                logger.info("File could not be splitted or resized")
                logger.exception('Error in resize, file could not be resized, skipping extraction')
                raise Exception('Error in resize, file could not be resized, skipping extraction')
            else:
                logger.info("PDF file does not need to be resized")
        except Exception as ex:
            logger.exception('Error in resize, PDF could not be resized, skipping extraction')
            raise ex

    def get_files_list(self):
        paths = []
        for subdir, dirs, files in os.walk(self.input_folder_path):
            for f in files:
                if f.endswith(".pdf") or f.endswith(".PDF"):
                    paths.append(os.path.join(subdir, f))
        logger.info("Number of files found {}".format(len(paths)))
        return paths

    def check_pdf_pages_size(self, original_file_path):
        max_size_allowed = self.sizes["max_with_allowed_pdf"] * self.sizes["max_height_allowed_pdf"]
        logger.info("Max size allowed: {}".format(str(max_size_allowed)))
        with fitz.open(original_file_path) as pdf_document:
            for (i, page) in enumerate(pdf_document):
                pix = page.get_pixmap(dpi=200)
                image_size_200_dpi = pix.width * pix.height
                logger.info("File {} page {} size after reading with 200 DPI: {}".format(os.path.basename(original_file_path),str(i), str(image_size_200_dpi)))
                if image_size_200_dpi > max_size_allowed:
                    del pix
                    return True
            del pix
            return False

    def split_pdf(self, original_file_path):
        try:
            files_path = []
            original_filename = os.path.basename(original_file_path.split(".")[0])
            inputpdf = Pdf.open(original_file_path)
            if not inputpdf.is_encrypted:
                self.doc_splitted_pdf_folder = os.path.join(self.temp_pdf_output_dir, original_filename + str(uuid.uuid4().hex))
                self.delete_create_temp_folder(self.doc_splitted_pdf_folder)
                for n, page in enumerate(inputpdf.pages):
                    doc_name = original_filename + "_page_" + str("%03d" % n) + ".pdf"
                    new_page_document_abs_path = os.path.join(self.doc_splitted_pdf_folder, doc_name)
                    dst = Pdf.new()
                    dst.pages.append(page)
                    dst.save(new_page_document_abs_path)
                    files_path.append(new_page_document_abs_path)
                    del dst
            return files_path
        except Exception as ex:
            logger.exception('Error in split_pdf - {0}'.format(ex))
            return []

    def check_image_size_resize(self, pdf_pages_paths, original_file_path):
        final_pdf_paths = []
        original_filename = os.path.basename(original_file_path.split(".")[0])
        self.doc_resized_splitted_pdf_folder = os.path.join(self.temp_pdf_resized_output_dir, original_filename + str(uuid.uuid4().hex))
        self.delete_create_temp_folder(self.doc_resized_splitted_pdf_folder)
        max_size_allowed = self.sizes["max_with_allowed_pdf"] * self.sizes["max_height_allowed_pdf"]
        for page_number, pdf_file_path in enumerate(pdf_pages_paths):
            with fitz.open(pdf_file_path) as pdf_document_fitz:
                pix = pdf_document_fitz[0].get_pixmap(dpi=200)
                image_size_200_dpi = pix.width * pix.height
                self.original_file_size = (pix.width, pix.height)
                logger.info("File {} page {} size after reading with 200 DPI: {}".format(os.path.basename(pdf_file_path), str(page_number), str(image_size_200_dpi)))
            if image_size_200_dpi > max_size_allowed:
                logger.info("File {} page {} need to be resized, current page size is {}, max size allowed is {}".format(os.path.basename(pdf_file_path), str(page_number), str(image_size_200_dpi), str(max_size_allowed)))
                resized_page_path = self.resize_pdf_page(pdf_file_path, page_number, original_file_path)
                final_pdf_paths.append(resized_page_path)
            else:
                logger.info("File {} does not need to be resized, current page size is {}".format(os.path.basename(pdf_file_path), str(image_size_200_dpi), str(max_size_allowed)))
                page_path = self.resize_pdf_page(pdf_file_path, page_number, resize=False)
                final_pdf_paths.append(page_path)
            del pix
        return final_pdf_paths

    def resize_pdf_page(self, path, page_number, original_file_path, resize=True):
        original_filename = os.path.basename(original_file_path.split(".")[0])
        filename = original_filename + "_page_{}.pdf".format(str("%03d" % page_number))
        pdf_page_path = os.path.join(self.doc_resized_splitted_pdf_folder, filename)
        if resize:
            with open(path, "rb") as f:
                pdf_file = PdfFileReader(f)
                new_page = pdf_file.getPage(0)
                original_page_width = float(new_page.mediaBox[2])
                original_page_height = float(new_page.mediaBox[3])
            highest_threshold_size = self.sizes["max_height_resized_pdf"] * self.sizes["max_width_resized_pdf"]
            selected_scale = self.get_scale(original_page_width, original_page_height, highest_threshold_size)
            with open(path, "rb") as f:
                pdf_file = PdfFileReader(f)
                new_page = pdf_file.getPage(0)
                new_page.scaleBy(selected_scale)
                writer = PdfFileWriter()
                writer.addPage(new_page)
                with open(pdf_page_path, "wb") as outfp:
                    writer.write(outfp)
                del writer
        else:
            shutil.copy(path, pdf_page_path)
        return pdf_page_path

    def get_scale(self, image_width, image_height, threshold):
        difference = None
        selected_scale = 1
        for scale in np.arange(0.1, 1, 0.05):
            size_by_scale = (image_height * scale) * (image_width * scale)
            new_difference = threshold - size_by_scale
            if new_difference > 0:
                if not difference or new_difference < difference:
                    selected_scale = scale
                    difference = new_difference
        logger.info("SCALE {}".format(selected_scale))
        return selected_scale

    def group_pdfs(self, pdf_paths, original_file_path):
        original_filename = os.path.basename(original_file_path.split(".")[0])
        self.doc_final_pdf_folder = os.path.join(self.final_pdf_output_dir, original_filename + str(uuid.uuid4()))
        self.create_folder(self.doc_final_pdf_folder)
        filename = original_filename + ".pdf"
        output_path = os.path.join(self.doc_final_pdf_folder, filename)
        pdf_concat = Pdf.new()
        for pdf_path in pdf_paths:
            src = Pdf.open(pdf_path)
            pdf_concat.pages.extend(src.pages)
        pdf_concat.save(output_path)
        del pdf_concat
        self.check_new_pdf_size(pdf_paths[-1])
        return output_path

    def delete_folders(self):
        folder_to_delete = [self.temp_pdf_output_dir, self.temp_pdf_resized_output_dir,
                            self.doc_splitted_pdf_folder, self.doc_resized_splitted_pdf_folder]
        for folder in folder_to_delete:
            if os.path.exists(folder):
                shutil.rmtree(folder)

    def delete_create_temp_folder(self, folder):
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)
        return folder

    def create_folder(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

    @staticmethod
    def change_working_directory():
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)

    def check_new_pdf_size(self, pdf):
        new_image = convert_from_path(pdf)
        logger.info("ORIGINAL FILE PAGE SIZE {}".format(self.original_file_size))
        logger.info("FINAL PDF PAGE SIZE {}".format(new_image[0].size))

if __name__ == '__main__':

    """
    pdf size  ------->  image resolution
    height x width      height x width
    59 x 42   ------->  160 x 120 (QQVGA)
    118 x 84  ------->  320 x 240 (QVGA)
    236 x 168 ------->  640 x 480 (VGA)
    295 x 211 ------->  800 x 600 (SVGA)
    377 x 270 ------->  1024 x 768 (XGA)
    424 x 304 ------->  1152 x 864 (XGA+)
    755 x 540 ------->  2048 x 1536 (QXGA)
    1180 x 844 -------> 3200 x 2400 (QUXGA)
    1510 x 1081 -------> 4096 x 3072 (HXGA)
    2360 x 1689 -------> 6400 x 4800 (HUXGA)


    """
    sizes = {
        #MAX PDF WIDTH AND HEIGHT, IF RESOLUTION IS ABOVE WIDTH*HEIGHT THE FILE WILL BE RESIZED
        "max_height_allowed_pdf": 295,
        "max_with_allowed_pdf": 211,

        #DESIRED PDF RESOLUTION IF FILE NEEDS TO BE RESIZED
        "max_height_resized_pdf": 295,
        "max_width_resized_pdf": 211,

        "max_image_pixels_allowed": 933120000
    }
    #FOLDER IN WHICH PDF FILES WILL BE SEARCHED TO BE RESIZED
    input_folder_path = "/home/alvaro/Downloads/5448_POP_documents"

    logger.info("RESIZING PROCESS STARTED")
    resize_pdf = ResizePdfFiles(input_folder_path, sizes)
    resize_pdf.resize()
    logger.info("RESIZING PROCESS FINISHED")