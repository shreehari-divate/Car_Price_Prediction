import logging
import sys

'''
 The below function takes two arguments one for error message and a sys object that contains info about the error
 It extracts the traceback object from the sys object, gets the filename and line number where the error occurred, and formats a detailed error message.
'''

def error_msg_detail(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_msg = "error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name,exc_tb.tb_lineno,str(error)
    )
    return error_msg

class CustomException(Exception):
    def __init__(self,error_msg,error_detail:sys):
        super().__init__(error_msg)
        self.error_msg = error_msg_detail(error_msg,error_detail=error_detail)

    def __str__(self):
        return self.error_msg

''''
 * class CustomException(Exception): This class defines a custom exception. It inherits from the built-in Exception class, which means it behaves like a standard Python exception but can also have additional features.
 * def __init__(self, error_msg, error_detail:sys): This is the constructor for the CustomException class. It takes an error message and a sys object as arguments, calls the error_msg_detail function to generate a detailed error message, and stores this message in the error_msg attribute.
 * def __str__(self): This method returns a string representation of the CustomException object. When you print a CustomException object or convert it to a string, this method will be called, and the detailed error message will be returned.
 '''
