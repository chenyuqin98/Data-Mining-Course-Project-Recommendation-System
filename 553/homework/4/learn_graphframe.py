
from graphframes import *
from pyspark import SparkContext
from pyspark.sql import SparkSession

# os.environ["PYSPARK_SUBMIT_ARGS"], = ("pyspark-shell --packages graphframes:graphframes:0.8.2-spark3.1-s_2.12")

def learn_graph():
  sc = SparkContext.getOrCreate()
  session = SparkSession(sc)
  vertices = session.createDataFrame([
    ("a", "Alice", 34),
    ("b", "Bob", 36),
    ("c", "Charlie", 30),
    ("d", "David", 29),
    ("e", "Esther", 32),
    ("f", "Fanny", 36),
    ("g", "Gabby", 60)], ["id", "name", "age"])
  edges = session.createDataFrame([
    ("a", "b", "friend"),
    ("b", "c", "follow"),
    ("c", "b", "follow"),
    ("f", "c", "follow"),
    ("e", "f", "follow"),
    ("e", "d", "friend"),
    ("d", "a", "friend"),
    ("a", "e", "friend")
  ], ["src", "dst", "relationship"])

  g = GraphFrame(vertices, edges)
  print(g)


def write():
  rlt = [['23y0Nv9FFWn_3UWudpnFMA'], ['3Vd_ATdvvuVVgn_YCpz8fw'], ['453V8MlGr8y61PpsDAFjKQ'], ['46HhzhpBfTdTSB5ceTx_Og'], ['Cf0chERnfd06ltnN45xLNQ'], ['F47atsRPw-KHmRVk5exBFw'], ['JeOHA8tW7gr-FDYOcPJoeA'], ['QYKexxaOJQlseGWmc6soRg'], ['SdXxLZQQnQNUEL1rGMOQ6w'], ['Si3aMsOVGSVlsc54iuiPwA'], ['YVQFzWm0H72mLUh-8gzd5w'], ['_m1ot2zZetDgjerAD2Sidg'], ['d5WLqmTMvmL7-RmUDVKqqQ'], ['eqWEgMH-DCP74i82BEAZzw'], ['gH0dJQhyKUOVCKQA6sqAnw'], ['gUu0uaiU7UEUVIgCdnqPVQ'], ['jJDUCuPwVqwjbth3s92whA'], ['jSbXY_rno4hYHQCFftsWXg'], ['tX0r-C9BaHYEolRUfufTsQ'], ['vENR70IrUsDNTDebbuxyQA'], ['0KhRPd66BZGHCtsb9mGh_g', '5fQ9P6kbQM_E0dx8DL6JWA'], ['98rLDXbloLXekGjieuQSlA', 'MJ0Wphhko2-LbJ0uZ5XyQA'], ['EY8h9IJimXDNbPXVFpYF3A', 'LiNx18WUre9WFCEQlUhtKA'], ['Gr-MqCunME2K_KmsAwjpTA', '_6Zg4ukwS0kst9UtkfVw3w'], ['QRsuZ_LqrRU65dTs5CL4Lw', 'lJFBgSAccsMGwIjfD7LMeQ'], ['S1cjSFKcS5NVc3o1MkfpwA', 'mm9WYrFhiNqvHCyhQKw3Mg'], ['750rhwO7D_Cul7_GtO9Jsg', 'DjcRgZ0cJbf6-W2TxvFlBA', 'MwpK7PqQX7fgTFM2Pfy61w'], ['9W73B44Iw8WslrTNB2CdCg', 'UmTMCfPlhA6kJLAsLycSfg', 'Uo5dPwoDpYBzOnmUnjxJ6A'], ['CLbpPUqP6XpeAfoqScGaJQ', 'drTMOo4p8nL0pnMNEyat2A', 'xhlcoVm3FOKcxZ0phkdO6Q'], ['SVC0CajvmYfH5uAq4JnGvg', 'ZZvfGGLnAkSBSUduV7KN-w', 'cyuDrrG5eEK-TZI867MUPA'], ['CyrRjt_7iJ8_lSHeH1_TlA', 'JhFK9D3LYl23Se3x4oPUxA', 'ZW-XoteNlRuuK-19q1spmw', 'lL-wNa0TKK6LXrlcVmjYrQ'], ['6YmRpoIuiq8I19Q8dHKTHw', 'XrRLaAeV20MRwdSIGjj2SQ', 'a48HhwcmjFLApZhiax41IA', 'angEr2YcXmCl20s8WQu32w', 'frQs7y5qa-X1pvAM0sJe1w'], ['2GUjO7NU88cPXpoffYCU8w', '6xi9tBoZ6r_v41u_XFsSnA', 'BDmxm7aeWFOLT35gSvkmig', 'H4EQn0rjFuGRgIm6c9NFLg', 'H5Asta4LpiKmRhSjWaogIg', 'e5sdXDOkCf0sIUAivXVluA'], ['903YwVSoAKyzudc8LH_HMA', '9S52XHEyrvOv4OZxU6pCLw', 'JM0GL6Dx4EuZ1mprLk5Gyg', 'JqjAthJThuVYgTh4iWDZ2A', 'XEqQG61fetXhuEV9RPslIA', 'ajxohdcsKhRGFlEvHZDyTw', 'hilL60vuuh06sMxs6Ckkog', 'nOTl4aPC4tKHK35T3bNauQ'], ['1st2ltGKJ00ZcRsev-Ieew', '4ZQq0ozRs-gXSz1z55iIDw', 'HLY9oDcVBH9D25lU4X_V5Q', 'Hv_q_ZnSIoZwdcoH0CyV2Q', 'Ih85YhFRDzOnB09yS__94g', 'KBoIRjxSW7OWczv8OS9Bew', 'LaiylSIbrA3aPvOYtl-J4A', 'Z9a1tDT8fVI75qXYwNhPpw', 'ZEq0WtRJD9Bl_vYgCsbfOg', 'e0Jn0ZjqL-dWi7Brs0bbmg', 'fOut10lknIp64tm3z6UTNg', 'l-1cva9rA8_ugLrtSdKAqA', 'oegRUjhGbP62M18WyAL6pQ', 'wXdrUQg4-VkSZH1FG4Byzw'], ['37HswRimgBEf7_US-c3CDA', 'DPtOaWemjBPvFiZJBi0m8A', 'IuaAfrkirlfzY3f4PkgSmw', 'MrsRJa4SWLq8XLU1RtPdlw', 'MtdSCXtmrSxj_uZOJ5ZycQ', 'PKEzKWv_FktMm2mGPjwd0Q', 'UwV6jBuTR1S9acT6bPTBPw', 'VdoTNYWuoXo01umgannw8A', 'WaAOt_eG0_-yLpG3fI--3g', 'cIbbfJEGLB3B-c8Po4AL5g', 'e5kg9bLvlJz-MEUrGjIeVQ', 'fcWM-oqjgS94yi1INhZa0g', 'm1IVpXClMox1VGw5hO2LhQ', 'xrvyW1ruKS0uz9RtFewC0Q'], ['2k8OVAPxlXHsA5X6EIoQpQ', '2xVrxhQJUBmOyG4ML77XKw', '5DgFmyjW6hkBtXtTMKl4tA', '7G8w2SnaC-qDVQ7_GqTxMg', '8oYMqhC5fhqAK_yxRjE7dQ', 'JRqMFKGxx6DnTGZrxwQZaA', 'NUtVG7jNPLJR2cxMXMH2-A', 'NlNlyQynkyEU3l7TR3LXdg', 'SsOiVav4V5_NjTl21Lj92w', 'TZ974xcbw2kqjYxAhDUYVg', 'UYcmGbelzRa0Q6JqzLoguw', 'Y0-lLNc2Y7gUGXPzSsMueQ', 'ZA1OT-PIZwz2kdHDA6mShw', 'Zk95TMXDx0zMUhYq5u8pxw', 'fLnkI1uHtXEsjtF6KoBHbQ', 'hLVq7VSJBHZwqurwWoCmpg', 'jcriwcTidug0fK8sgAloHA', 'tekHDsd0fskYG3tqu4sHQw'], ['0gZ8E5tBWTEtGEZDuTzhzw', '4ONcRRisDZkbV1cviA7nFw', '9SWtEX1k9AjRg93BAzMCpg', 'A-U-K9z9oraMH7eBZW1dOA', 'B7IvZ26ZUdL2jGbYsFVGxQ', 'BE4fE4R3TaVn8xy4sYYjbg', 'DgfsJqg_gozVgaeZ5vjllA', 'EI9ijI9Wh66LrVW-GmWkOg', 'FyQrUamokaPLDrBxGmzPnA', 'Gua5GdTlTWJpovtG7Hdtyg', 'LKP0Yq9T7Ss6oiDZnVtQwQ', 'LgFDWZTLi1w9OGi5BtKORg', 'Nf_Jw_W_CwOz5WJ7ApSMxg', 'ORJnGXXkS9tQBTNyPQJF9A', 'QUYbGl1DL-9faG150MQ7zA', 'QvLg2kxqHHahxxOlHlEIZw', 'SX_SMrddkDU5dySbsZMu9A', 'WXlxViTwXHPBvhioljN9PQ', 'YA-caxALI4C-eCiSM97new', 'ZXyGw3Z1DyhK1sfNtpcyYA', 'ae7zi8F0B6l_JCITh1mXDg', 'h-ajC_UHD0QAyAzySN6g2A', 'jnn504CkjtfbYIwBquWmBw', 'k24kSTpZHUdEd-QYXLy3fQ', 'k58KNO8Rya-q8njKq8-uBQ', 'o-t-i7nbT5N_cmkCXs5oDQ', 'pDNeS1nbkKS7mJmhRQJPig', 'tcWnoX_IfuDmlDl6o6y3_g'], ['0FMte0z-repSVWSJ_BaQTg', '0FVcoJko1kfZCrJRfssfIA', '0QREkWHGO8-Z_70qx1BIWw', '1KQi8Ymatd4ySAd4fhSfaw', '23o7tyUGlC6FCDVcyqLeFA', '2XYdguaaZ7dgi6fAlddujg', '2quguRdKBzul3GpRi9e1mA', '39FT2Ui8KUXwmUt6hnwy-g', '4PQhC-zTQ4ACEN0-r39JuQ', '4pc_EyanaC3ARh0MZZyouA', '79yaBDbLASfIdB-C2c8DzA', '7RCz4Ln_FaTvNrdwe251Dg', '7Vfy39A_totC-w70qZi0MA', '97j2wkFU46OOgm6ErRAb7w', '9xM8upr_n9jchUDKxqSGHw', 'Ams0iLRd0AhZZectGKA8fw', 'B0ENvYKQdNNr1Izd2r-BAA', 'BDjiEmXljD2ZHT61Iv9rrQ', 'CebjpVd3PsofCgotWp60pg', 'ChshgCKJTdIDg17JKtFuJw', 'DKolrsBSwMTpTJL22dqJRQ', 'DkLSyxogCcJXY5DbTZ-f2A', 'ELfzWgdf64VBLi5z1ECItw', 'EiwxlbR8fb68lMgEXhcWKA', 'IXD-jdycm7m34b_Nliy82g', 'JLv2Dmfj73-I0d9N41tz1A', 'JteQGisqOf_pklq7GA0Rww', 'KHjroLTG6Ah8LyItTyB2yw', 'KLB3wIYUwKDPMbijIE92vg', 'KgJdBWS3ReP6TVhYWJRKmg', 'KtE55izPs1ubJn3ofF2IrA', 'LcCRMIDz1JgshpPGYfLDcA', 'O9pMFJSPg80YVzpMfNikxw', 'OoyQYSeYNyRVOmdO3tsxYA', 'PE8s8ACYABRNANI-T_WmzA', 'R4l3ONHzGBakKKNo4TN9iQ', 'S9dDf0JqSMAvusp5f-9bGw', 'T88y73qdOSutuvzLlhWtqQ', 'TjsBbWAfwxWEXPxaLNv5SQ', 'Tk_FWXueutKii3f9yJFsdw', 'UAB1Zyg6Q0oEpXeYRf5K_g', 'WoKCLSctS7G2547xKcED-Q', 'XPAJ2KHkCwBA0vafF-2Zcg', 'XUEwSGOGARxW-3gPiGJKUg', '_Pn-EmWO-pFPFg81ZIEiDw', '_VTEyUzzH92X3w-IpGaXVA', 'ay4M5J28kBUf0odOQct0BA', 'bE7Yd0jI_P6g27MWEKKalA', 'bHufZ2OTlC-OUxBDRXxViw', 'bJguBxPlnTW29tRTAF0nkQ', 'bSUS0YcvS7UelmHvCzNWBA', 'bbK1mL-AyYCHZncDQ_4RgA', 'cm3_8c_NDhPcpwJQ96Aixw', 'dTeSvET2SR5LDF_J07wJAQ', 'dW6bAWM1HbPdk_cGS_a2HA', 'dzJDCQ5vubQBJTfYTEmcbg', 'e8uzNcSC5tQMD22GNAQEQA', 'hd343st7cOIUSfAd5r0U7A', 'hqmnMdDS-Opjp3BfBJA8qA', 'j8Dts8irvVBwEhEEae_-wA', 'jPcrABeWgWlTPi-E0Op_aA', 'jgoG_hHqnhZvQEoBK0-82w', 'kKTcYPz47sCDH1_ylnE4ZQ', 'kwIhn1_cnQeUaLN0CuWWHw', 'ma6206bmu-a_Ja7Iv-yRCw', 'mnoe2vwouRADn97dTDkw4A', 'mu4XvWvJOb3XpG1C_CHCWA', 'p9942XebvxZ9ubHm4SXmMQ', 'qd16czwFUVHICKF7A4qWsQ', 'qtOCfMTrozmUSHWIcohc6Q', 'sBqCpEUn0qYdpSF4DbWlAQ', 'sO6iNKgv_ToVfof-aQWgXg', 'sdLns7062kz3Ur_b8wgeYw', 'tAcY4S3vIuNlAoRlCcz5VA', 'tL2pS5UOmN6aAOi3Z-qFGg', 'tRZAC_H5RHrjvyvtufcNXQ', 'voXU5A3FfOcXZ2VNsJ0q4w', 'waN6iwcphiVEoCews4f4CA', 'y6jsaAXFstAJkf53R4_y4Q', 'yCaDISH0R8e5U376zDWTpQ', 'zBi_JWB5uUdVuz3JLoAxGQ']]
  with open('test.txt', 'w') as f:
    for r in rlt:
      f.writelines(str(r)[1:-1] + '\n')

if __name__ == '__main__':
  write()

  #   r = ['0FVcoJko1kfZCrJRfssfIA','0FVcoJko1kfZCrJRfssfIA',],
  #   f.writelines(str(r)[1:-1], + '\n')

