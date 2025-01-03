from colorama import init, Fore, Style

# Initialize colorama for colored console output
init(autoreset=True)

def print_colored(message, color=Fore.WHITE, emoji=""):
    """
    Prints a message to the console with specified color and an optional emoji.

    Parameters:
        message (str): The message to print.
        color (str): The color to display the message in (default is white).
        emoji (str): An optional emoji to add at the beginning (default is none).
    """
    print(f"{color}{emoji}  {message} {Style.RESET_ALL}")

def highlight_value(value, good_threshold=0.8, bad_threshold=0.5):
    """
    Highlights values above a certain threshold in green for good performance and
    below a certain threshold in red for bad performance.
    
    Parameters:
        value (float): The value to evaluate.
        good_threshold (float): The threshold above which the value is considered good (default is 0.8).
        bad_threshold (float): The threshold below which the value is considered bad (default is 0.5).
    
    Returns:
        str: A colored value string.
    """
    if value >= good_threshold:
        return f"{Fore.GREEN}{value:.2f}{Style.RESET_ALL}"  # Green for good values
    elif value <= bad_threshold:
        return f"{Fore.RED}{value:.2f}{Style.RESET_ALL}"  # Red for bad values
    else:
        return f"{value:.2f}"  # Normal formatting for neutral values