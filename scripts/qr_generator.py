# import qrcode


# qr = qrcode.QRCode(version=1,
#                    error_correction=qrcode.constants.ERROR_CORRECT_L,
#                    box_size=20,
#                    border=1)

# qr.add_data("Chutiya hai tu")
# qr.make(fit=True)

# img = qr.make_image(fill_color="black", back_color="white")
# img.save("advance.png")


# import qrcode

# def generate_qr(data, file_path="qr_code.png", box_size=10, border=4, fill_color="black", back_color="white"):
#     """
#     Generates a QR code image and saves it to the specified file path.
    
#     Parameters:
#     - data (str): The data to encode in the QR code.
#     - file_path (str): The output file path for the saved QR code image.
#     - box_size (int): Size of each box in the QR code.
#     - border (int): The width of the border (minimum is 4).
#     - fill_color (str): Color of the QR code.
#     - back_color (str): Background color of the QR code.
#     """
#     qr = qrcode.QRCode(
#         version=1,
#         error_correction=qrcode.constants.ERROR_CORRECT_L,
#         box_size=box_size,
#         border=border
#     )

#     qr.add_data(data)
#     qr.make(fit=True)

#     img = qr.make_image(fill_color=fill_color, back_color=back_color)
#     img.save(file_path)
#     print(f"QR code saved to {file_path}")

# # Example usage
# generate_qr("Hello, World!", file_path="hello_world_qr.png", box_size=20, border=1)






import qrcode
import requests

def get_current_location():
    """
    Retrieves the current geographic location using the free API from ipinfo.io.
    Returns a string with latitude and longitude or 'Unknown location' if failed.
    """
    try:
        # Fetch location data from ipinfo.io
        response = requests.get("https://ipinfo.io", timeout=10)
        response.raise_for_status()
        location_data = response.json()

        # Extract location (latitude and longitude) and city
        loc = location_data.get("loc", "0,0").split(",")
        city = location_data.get("city", "Unknown city")
        country = location_data.get("country", "Unknown country")
        latitude, longitude = loc[0], loc[1]

        return f"{latitude}, {longitude} ({city}, {country})"
    except Exception as e:
        print("Error fetching location:", e)
        return "Unknown location"

def generate_qr(data, file_path="qr_code.png", box_size=10, border=4, fill_color="black", back_color="white"):
    """
    Generates a QR code image and saves it to the specified file path.
    
    Parameters:
    - data (str): The data to encode in the QR code.
    - file_path (str): The output file path for the saved QR code image.
    - box_size (int): Size of each box in the QR code.
    - border (int): The width of the border (minimum is 4).
    - fill_color (str): Color of the QR code.
    - back_color (str): Background color of the QR code.
    """
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=box_size,
        border=border
    )

    qr.add_data(data)
    qr.make(fit=True)

    img = qr.make_image(fill_color=fill_color, back_color=back_color)
    img.save(file_path)
    print(f"QR code saved to {file_path}")

# Fetch current location
location = get_current_location()

# Generate a QR code with location data
generate_qr(f"Current Location: {location}", file_path="location_qr.png", box_size=20, border=1)
