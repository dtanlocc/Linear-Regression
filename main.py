import sys
from PyQt5 import QtWidgets, uic

class Gui():
    def __init__(self):
        self.main_win = QtWidgets.QMainWindow()
        self.ui = uic.loadUi('main.ui',self.main_win)
        self.ui.predict.clicked.connect(self.clicked)
    def show(self):
        self.main_win.show()
        
    def clicked(self):
        result = linear_regression.predict([[int(self.ui.GRE_Score.toPlainText()),int(self.ui.TOEFL_Score.toPlainText()),int(self.ui.rate.currentText()),float(self.ui.SOP.currentText()),float(self.ui.LOR.currentText()),float(self.ui.CGPA.toPlainText()),int(self.ui.research.currentText())]])
        self.ui.view.setText("Chance of Admit Predict"+str(result)[0:-1])
app = QtWidgets.QApplication(sys.argv)
f1 = Gui()
f1.show()
app.exec()