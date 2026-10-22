"""Check dashboard navigation, filters and prediction state with real artifacts."""
import unittest
from datetime import date
from streamlit.testing.v1 import AppTest
from module2 import ROOT, CLASS_NAMES


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(str(ROOT/'app.py'), default_timeout=30).run()
        self.assertFalse(self.app.exception)

    def go(self, page):
        self.app.radio(key='page').set_value(page).run()
        self.assertFalse(self.app.exception)

    def test_pages_and_filters(self):
        self.assertEqual(self.app.metric[0].value, '420,768')
        self.go('Khám phá dữ liệu')
        self.app.multiselect(key='eda_stations').set_value(['Aotizhongxin','Changping']).run()
        self.assertFalse(self.app.exception)
        self.app.selectbox[0].set_value('Test').run()
        self.app.selectbox[1].set_value('TEMP').run()
        self.assertFalse(self.app.exception)
        observations = self.app.dataframe[-1].value
        self.assertEqual(len(observations), int(self.app.metric[0].value.replace(',', '')))
        self.assertGreater(len(observations), 500)
        self.app.multiselect(key='eda_stations').set_value([]).run()
        self.assertFalse(self.app.exception)
        self.assertTrue(any('Chọn ít nhất' in x.value for x in self.app.info))
        self.go('Nhãn & mô hình')
        self.app.selectbox[0].set_value('Cao').run()
        self.app.radio(key='coefficient_group').set_value('Biến phân loại one-hot').run()
        self.assertFalse(self.app.exception)
        self.go('Thực nghiệm & đánh giá')
        self.app.selectbox[0].set_value('validation').run()
        self.app.radio(key='evaluation_scope').set_value('observed_current').run()
        self.app.selectbox[1].set_value('month').run()
        self.app.checkbox[0].check().run()
        self.assertFalse(self.app.exception)
        errors = self.app.dataframe[-1].value
        self.assertGreater(len(errors), 300)
        self.assertTrue((errors.label != errors.logistic).all())
        self.assertEqual(self.app.metric[0].value, '102,479')
        self.go('Báo cáo & bàn giao')

    def test_prediction_and_changed_input(self):
        self.go('Dự đoán tương tác')
        self.app.button(key='predict').click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.metric[0].label, 'Nhóm dự đoán')
        self.assertIn(self.app.metric[0].value, CLASS_NAMES)
        self.assertEqual(len(self.app.dataframe[-1].value), 31)
        self.app.selectbox(key='prediction_station').set_value('Changping').run()
        self.assertFalse(self.app.exception)
        self.assertNotIn('Nhóm dự đoán', [m.label for m in self.app.metric])
        self.app.date_input[0].set_value(date(2016,2,29)).run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.selectbox[1].options, ['23'])
        self.app.button(key='predict').click().run()
        self.assertFalse(self.app.exception)
        self.assertIn(self.app.metric[0].value, CLASS_NAMES)


if __name__ == '__main__':
    unittest.main()
