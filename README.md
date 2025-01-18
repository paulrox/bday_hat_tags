# Tags with Birthday Hat

## Install

Optional: Create a virtual environment before installing the requirements

```
pip install -r requirements.txt
```

## Usage

```
python bday_hat_tags.py
```

Export to PDF with tags width of 2 centimeters:
```
python export_to_pdf.py output/ output.pdf --rows 5 --cols 5 --image_width 236
```

Export to PDF with tags width of 3 centimeters:
```
python export_to_pdf.py output/ output.pdf --rows 4 --cols 4 --image_width 354
```