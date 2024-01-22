import os

class ProjConfig():
    def __init__(self):
        # Default configuration values
        self.baseDir = '/content/drive/MyDrive/TransferLearningProject'
        self.baseDirPelvic = os.path.join(self.baseDir,'PelvicMRData/data')
        self.baseDirAmos22 = os.path.join(self.baseDir,'amos22/amos22')


config = ProjConfig()