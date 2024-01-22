import os

class ProjConfig():
    def __init__(self):
        # Default configuration values
        self.baseDir = './baseDir'

        # those are based on the specified folder structure in instructions.pdf
        # change these for other environments
        self.baseDirPelvic = os.path.join(self.baseDir,'PelvicMRData/data')
        self.baseDirAmos22 = os.path.join(self.baseDir,'amos22/amos22')


config = ProjConfig()