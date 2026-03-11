#!/usr/bin/env python3
# pip3 install tqdm PyMuPDF

import hashlib
import pathlib
import time
import urllib.error
import urllib.request

import fitz
import tqdm.auto


PDF_DIR = pathlib.Path("pdf")
PNG_DIR = pathlib.Path("png")


def download():
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    for k, url, checksum in tqdm.auto.tqdm(PDF_LIST, desc="Download"):
        pdf_file = PDF_DIR / f"{k}.pdf"

        if pdf_file.exists():
            h = hashlib.md5()
            with open(pdf_file, "rb") as f:
                for chunk in iter(lambda: f.read(1048576), b""):
                    h.update(chunk)
            if not checksum or h.hexdigest() == checksum:
                continue

        cnt = 0
        while True:
            time.sleep(2)
            try:
                h = hashlib.md5()
                with urllib.request.urlopen(url) as r, open(pdf_file, "wb") as w:
                    while True:
                        chunk = r.read(1048576)
                        if not chunk:
                            break
                        w.write(chunk)
                        h.update(chunk)
                break
            except urllib.error.URLError as e:
                time.sleep(60)
                cnt += 1
                if cnt >= 5:
                    raise e

        if checksum and h.hexdigest() != checksum:
            raise ValueError(f"{pdf_file}: FAILED")


def extract():
    PNG_DIR.mkdir(parents=True, exist_ok=True)

    for k, _, _ in tqdm.auto.tqdm(PDF_LIST, desc="Extract"):
        pdf_file = PDF_DIR / f"{k}.pdf"
        with fitz.open(pdf_file) as doc:
            for page_num in range(len(doc)):
                pix = doc.load_page(page_num).get_pixmap(dpi=150)
                pix.save(PNG_DIR / f"{pdf_file.stem}_page_{page_num + 1}.png")


def main():
    download()
    extract()


PDF_LIST = [
    ("01-020-001", "https://web.archive.org/web/20251216113054if_/https://www.sorachi.pref.hokkaido.lg.jp/fs/7/6/5/7/8/8/5/_/%E6%A0%BD%E5%9F%B9%E6%9A%A6.pdf", "7ab9009b450ae2a992454a9b56237e92"),
    ("02-021-001", "https://web.archive.org/web/20250504083108if_/https://ja-goshotsugaru.com/assets/images/2025/03/suitogoyomi2025.pdf", "80a9f9438b9f0adc4150d7ebe9a3dcfc"),
    ("02-021-002", "https://web.archive.org/web/20240405171919if_/https://ja-goshotsugaru.com/assets/images/2023/09/ringo.pdf", "ab1c9b5b38e1e022a51cc1a95e8b4cf5"),
    ("03-009-001", "https://web.archive.org/web/20251216113112if_/https://www.jahanamaki.or.jp/imagem/saibai/1494923606_01.pdf", "fd91b61a5c5086d2cc5ddf2867f7d6c2"),
    ("03-009-002", "https://web.archive.org/web/20251216113112if_/https://www.jahanamaki.or.jp/rice/koyomi/files/h30_koyomi.pdf", "c2e38500855ebc2fb3fc34c4c889b4ec"),
    ("03-009-003", "https://web.archive.org/web/20220720120432if_/https://www.jahanamaki.or.jp/imagem/saibai/1494923519_01.pdf", "593c8048f4832d2877338f344fbb8a8e"),
    ("03-009-004", "https://web.archive.org/web/20251216113114if_/https://www.jahanamaki.or.jp/rice/koyomi/files/h30_koyomi2.pdf", "c50dfafcb5086dc6b131ca3a8ebc8681"),
    ("03-009-005", "https://web.archive.org/web/20251216113116if_/https://www.jahanamaki.or.jp/rice/koyomi/files/h30_koyomi4.pdf", "68020bd7ebda7ad8f3e2a473580b1e39"),
    ("03-009-006", "https://web.archive.org/web/20251217225919if_/www.jahanamaki.or.jp/rice/koyomi/files/h30_koyomi.pdf", "c2e38500855ebc2fb3fc34c4c889b4ec"),
    ("03-009-007", "https://web.archive.org/web/20251225180405if_/www.jahanamaki.or.jp/rice/koyomi/files/h30_koyomi2.pdf", "c50dfafcb5086dc6b131ca3a8ebc8681"),
    ("03-009-009", "https://web.archive.org/web/20251216113116if_/www.jahanamaki.or.jp/rice/koyomi/files/h30_koyomi4.pdf", "68020bd7ebda7ad8f3e2a473580b1e39"),
    ("03-009-010", "https://web.archive.org/web/20251217231718if_/www.jahanamaki.or.jp/rice/koyomi/files/h30_koyomi5.pdf", "25f1353f92adc35219364325625f48d4"),
    ("03-022-001", "https://web.archive.org/web/20240814210756if_/https://ja-iwatechuoh.or.jp/contents/wp-content/uploads/2024/01/r3hitomebore-koyomi.pdf", "17d69e555c1e8f9f6b88e6633a3c1665"),
    ("03-022-002", "https://web.archive.org/web/20241205093248if_/https://ja-iwatechuoh.or.jp/contents/wp-content/uploads/2024/01/r4himenomochi-koyomi.pdf", "e3d8c8c74d6d028046ec0cbaad32a32b"),
    ("04-025-001", "https://web.archive.org/web/20240626082452if_/https://www.ja-shinmiyagi.or.jp/wp/wp-content/uploads/2020/02/b57e5e1d231309b650e5ff4be829c97a.pdf", "02dee82d88292e2908bcfcf37217463b"),
    ("04-025-002", "https://web.archive.org/web/20240626112713if_/https://www.ja-shinmiyagi.or.jp/wp/wp-content/uploads/2021/03/32330b800402b1aa9ae2ead1eca06bfa.pdf", "c96c137477c9c2695356b2ee6e242ba9"),
    ("04-025-003", "https://web.archive.org/web/20240518004527if_/https://www.ja-shinmiyagi.or.jp/wp/wp-content/uploads/2022/03/0c0b065aa81f40cb2678f9bad4b59546.pdf", "c30cca54ba56dbcd8540318603bc6c9b"),
    ("05-023-001", "https://web.archive.org/web/20250221043607if_/https://www.pref.akita.lg.jp/uploads/public/archive_0000073119_00/%E3%81%82%E3%81%8D%E3%81%9F%E3%81%93%E3%81%BE%E3%81%A1R%E6%A0%BD%E5%9F%B9%E6%9A%A6.pdf", "7c702e3bc397b1fb07d04b5d826642bf"),
    ("06-004-001", "https://web.archive.org/web/20251224061720if_/www.midorinet.or.jp/overview/wp-content/uploads/sites/4/2025/02/kabu.pdf", "72e58c76d3cc5be803cd37ccf2b7eae4"),
    ("06-004-002", "https://web.archive.org/web/20251229175837if_/www.midorinet.or.jp/overview/wp-content/uploads/sites/4/2024/12/asuparagasumanyu.pdf", "1d2e528df8a435f19fb102e714a4be84"),
    ("06-004-005", "https://web.archive.org/web/20251217171308if_/www.midorinet.or.jp/overview/wp-content/uploads/sites/4/2025/02/edamame.pdf", "9af745eed268c128ce69c9437e34ca55"),
    ("06-004-006", "https://web.archive.org/web/20251227103414if_/www.midorinet.or.jp/overview/wp-content/uploads/sites/4/2025/02/himawari.pdf", "d9c27a40cd5cfd0bad878b2760685046"),
    ("06-004-007", "https://web.archive.org/web/20251228123437if_/https://www.midorinet.or.jp/overview/wp-content/uploads/sites/4/2025/02/hourensou.pdf", "3d3766a0018877b19c3fbeeba224549c"),
    ("06-004-008", "https://web.archive.org/web/20260106212422if_/https://www.midorinet.or.jp/overview/wp-content/uploads/sites/4/2025/02/ishokust.pdf", "ca38b81041bd23f390f8060b1d1ea312"),
    ("06-004-010", "https://web.archive.org/web/20251220115313if_/https://www.midorinet.or.jp/overview/wp-content/uploads/sites/4/2025/02/jikamakist.pdf", "f314dfc953ca5f68801c98ebafc48900"),
    ("06-004-012", "https://web.archive.org/web/20251224061720if_/https://www.midorinet.or.jp/overview/wp-content/uploads/sites/4/2025/02/kabu.pdf", "72e58c76d3cc5be803cd37ccf2b7eae4"),
    ("06-004-013", "https://web.archive.org/web/20251222001124if_/https://www.midorinet.or.jp/overview/wp-content/uploads/sites/4/2025/02/keitou.pdf", "f45a26d86f4141b7925a20bc07092fdd"),
    ("06-004-014", "https://web.archive.org/web/20251216182333if_/https://www.midorinet.or.jp/overview/wp-content/uploads/sites/4/2025/02/kodamasuika.pdf", "bd2340ca88d2f37ba3a3dd71407c4ab5"),
    ("06-004-017", "https://web.archive.org/web/20260103000623if_/https://www.midorinet.or.jp/overview/wp-content/uploads/sites/4/2025/02/minitomato.pdf", "8db7fcf921c18794ad22450b908a55e8"),
    ("06-004-018", "https://web.archive.org/web/20251218112617if_/https://www.midorinet.or.jp/overview/wp-content/uploads/sites/4/2025/02/mizuna.pdf", "e809a048d224909b14aa6054040793f5"),
    ("06-007-025", "https://web.archive.org/web/20251221232908if_/https://agrin.jp/documents/1127/file1_file0117112213552223943.pdf", "54b22891a78f0c21f630f3269cd955f2"),
    ("06-007-026", "https://web.archive.org/web/20240720115615if_/https://agrin.jp/documents/1127/file1_file0117112214014824658.pdf", "251c900ee11a5d6b33837df4b7aee281"),
    ("06-007-027", "https://web.archive.org/web/20240720132724if_/https://agrin.jp/documents/1127/file1_file0117112214021724748.pdf", "7009192613a63da7f906ecd255bf517a"),
    ("06-007-028", "https://web.archive.org/web/20251220030357if_/https://agrin.jp/documents/1127/file1_file0117112214033124922.pdf", "22ff49154a13467e7f28c051d022921e"),
    ("06-007-029", "https://web.archive.org/web/20251216185156if_/https://agrin.jp/documents/1127/file1_file0117112214035824975.pdf", "636d0745ff8a2772d680f40dc2902bce"),
    ("06-007-030", "https://web.archive.org/web/20240720120027if_/https://agrin.jp/documents/1127/file1_file0117112214042025061.pdf", "11091b73dbd00f069715600d321ab99c"),
    ("06-007-031", "https://web.archive.org/web/20251218135458if_/https://agrin.jp/documents/1127/file1_file0117112214044825099.pdf", "bfc22d21b950694f8f56915fb3eedcf8"),
    ("06-007-032", "https://web.archive.org/web/20240720114946if_/https://agrin.jp/documents/1127/file1_file0117112214050525191.pdf", "a362eab21ae8b5c5c6d1f802cf6196ee"),
    ("06-007-033", "https://web.archive.org/web/20251216225227if_/https://agrin.jp/documents/1127/file1_file0117112214054425304.pdf", "180e3cfaf8645ae5ee4130e54561f166"),
    ("06-007-034", "https://web.archive.org/web/20251226225740if_/https://agrin.jp/documents/1127/file1_file0117112214061025381.pdf", "29436540442c881e82171aab171dedfb"),
    ("06-007-035", "https://web.archive.org/web/20240720122231if_/https://agrin.jp/documents/1127/file1_file0118082819540911477.pdf", "79dfe0d03222d192141959731b2c9f03"),
    ("06-007-036", "https://web.archive.org/web/20251223163203if_/https://agrin.jp/documents/1127/file1_file0118082819551111618.pdf", "c600329999ec602ce14c0bd8b2fbee0c"),
    ("06-007-037", "https://web.archive.org/web/20251217110110if_/https://agrin.jp/documents/1127/file1_file0118082819554611763.pdf", "e1a236aa70bb66fe9bbd3dc773568e3f"),
    ("07-026-001", "https://web.archive.org/web/20250117184603if_/http://aizumiyakawa.jp/inasakuR3.99.pdf", "38490b1e68391b32f234814d0be44409"),
    ("07-026-002", "https://web.archive.org/web/20251216113132if_/http://aizumiyakawa.jp/inasakuR3.98.pdf", "f04044537b65b18841fb22931b267398"),
    ("07-027-001", "https://web.archive.org/web/20250908103221if_/https://pref.fukushima.lg.jp/uploaded/attachment/334601.pdf", "6724ee71ea77e39b2250c20a3354dcd9"),
    ("07-027-002", "https://web.archive.org/web/20250908101636if_/https://pref.fukushima.lg.jp/uploaded/attachment/334600.pdf", "f1fe82fd6187db76aebb4326b5296be9"),
    ("07-027-003", "https://web.archive.org/web/20250326110803if_/http://www.pref.fukushima.lg.jp/uploaded/attachment/670940.pdf", "27e2e35391a65f3b36a45f296a5ce7cd"),
    ("07-028-001", "https://web.archive.org/web/20251216113138if_/https://www.pref.fukushima.lg.jp/uploaded/attachment/683115.pdf", "21754f77c1065b54e1cb879931d7530b"),
    ("07-028-002", "https://web.archive.org/web/20251216113145if_/https://www.pref.fukushima.lg.jp/uploaded/attachment/683114.pdf", "49a037b485cfb138f85160ea70308b24"),
    ("08-029-001", "https://web.archive.org/web/20240425020810if_/https://www.ibanourin.or.jp/cms/wp-content/uploads/2016/05/fc8116d46a2c73667466c3ccce7e9614.pdf", "5122c0c8a2a3c1876006c5242ec3fc3e"),
    ("08-029-002", "https://web.archive.org/web/20241009201921if_/https://www.ibanourin.or.jp/cms/wp-content/uploads/2017/10/27b32263b59016dbc13514d4a383b2a0.pdf", "f4d3a5cccfca1b124aac45532f8618f2"),
    ("08-029-003", "https://web.archive.org/web/20250615172755if_/https://www.ibanourin.or.jp/cms/wp-content/uploads/2016/03/soba_koyomi.pdf", "283df9a8389d462e3eb344b0b1e2fe19"),
    ("08-029-004", "https://web.archive.org/web/20240719114707if_/https://www.ibanourin.or.jp/cms/wp-content/uploads/2016/10/35fbcf000616ae83d53897899d5d76ab.pdf", "d42e6de9e227e671178160bd4e2297fd"),
    ("08-029-005", "https://web.archive.org/web/20240619191531if_/http://www.ibanourin.or.jp/cms/wp-content/uploads/2016/03/koshihikari_koyomi.pdf", "af827c5ac563d3dc7ae58a3d4c94766c"),
    ("08-030-001", "https://web.archive.org/web/20251216113205if_/https://www.pref.ibaraki.jp/nourinsuisan/noken/documents/satonosora_manual_2015_03.pdf", "cc06f4e4994447e0e8af0f4c9eba4506"),
    ("08-030-002", "https://web.archive.org/web/20251216113213if_/https://www.pref.ibaraki.jp/nourinsuisan/noken/documents/kashimagoal_manual_2015_3.pdf", "dc6944bc65e41c8113cf8e998387fc2b"),
    ("08-031-001", "https://web.archive.org/web/20240723011616if_/https://www.ibaraki-suiden.jp/wp-content/uploads/2024/02/chihominori3.pdf", "7948b16f4d20d8f3c1b4e8d085859d7c"),
    ("08-031-002", "https://web.archive.org/web/20240718165328if_/https://www.ibaraki-suiden.jp/wp-content/uploads/2023/09/akidawara.pdf", "f9ecdc63b136ece073d0fe296c9a738b"),
    ("08-031-003", "https://web.archive.org/web/20240720071431if_/https://www.ibaraki-suiden.jp/wp-content/uploads/2023/09/tsukinohikari.pdf", "02cc582c3c782a656ba0b72f4c6f7abd"),
    ("08-031-004", "https://web.archive.org/web/20240718150007if_/https://www.ibaraki-suiden.jp/wp-content/uploads/2023/09/yumeaoba.pdf", "6cdd926e95da1fd16eb2b3d68e9668de"),
    ("08-031-005", "https://web.archive.org/web/20250524094802if_/https://www.city.tsukuba.lg.jp/material/files/group/107/saibaireki.pdf", "98b671afa79a2ec61f616d9e2ab481b7"),
    ("09-045-001", "https://web.archive.org/web/20241011050152if_/https://tochigi-beibaku.or.jp/data/gijutsu/yumeaoba.pdf", "93801b17260dea62d10e89e5abf7cdc1"),
    ("09-045-002", "https://web.archive.org/web/20251216113242if_/https://www.tochigi-beibaku.or.jp/data/syushi/tochiginohoshi-saibaigoyomi-hayaue.pdf", "4340c629455ac7f77b54635977399a0c"),
    ("09-046-001", "https://web.archive.org/web/20251216113244if_/https://www.pref.tochigi.lg.jp/g55/keieihukyubu/r6/noutiku/documents/20241108140017.pdf", "e0dd1bec514221adcd96ed4eb6170570"),
    ("09-046-002", "https://web.archive.org/web/20251216113246if_/https://www.pref.tochigi.lg.jp/g55/keieihukyubu/r6/noutiku/documents/20241108140042.pdf", "02dc846ae251137ba5aefe6151cc5db0"),
    ("09-046-003", "https://web.archive.org/web/20251216134100if_/https://www.pref.tochigi.lg.jp/g55/keieihukyubu/r6/noutiku/documents/20241108140104.pdf", "36b8891667de5d11e5a0064ef9ef5739"),
    ("10-047-001", "https://web.archive.org/web/20250124021137if_/http://www.jakantomi.or.jp/wp-content/uploads/SpecialCultivationTamanegiJyoho_H24.pdf", "6e1417f3fcd3987de4e52426627bd663"),
    ("10-047-002", "https://web.archive.org/web/20250124033604if_/http://www.jakantomi.or.jp/wp-content/uploads/SpecialCultivationNiraJyoho.pdf", "7684de0dc0dd4adc9bd66169772e4de7"),
    ("11-048-001", "https://web.archive.org/web/20251216113309if_/https://www.pref.saitama.lg.jp/documents/3354/29satonohohoemi.pdf", "cba386ca30de2d7b19d27e3c2623dbff"),
    ("11-049-001", "https://web.archive.org/web/20241003160557if_/https://www.pref.saitama.lg.jp/documents/104573/satoimosaibaireki.pdf", "fe1c1679f6fb50de546807efeee51749"),
    ("11-050-001", "https://web.archive.org/web/20240613170530if_/https://www.pref.saitama.lg.jp/documents/70308/gomasaibaigoyomi.pdf", "5b326fbb7bf503b0dd38b4f2d08481cd"),
    ("12-001-001", "https://web.archive.org/web/20251216142451if_/www.zennoh.or.jp/cb/producer/einou/calendar/pdf/broccoli.pdf", "d52df2209e951adf3f0a0a5cb0479997"),
    ("12-001-002", "https://web.archive.org/web/20251219165142if_/www.zennoh.or.jp/cb/producer/einou/calendar/pdf/cabbage.pdf", "48c743b6332d81161d2f24a1dcfc4611"),
    ("12-001-003", "https://web.archive.org/web/20251220115012if_/www.zennoh.or.jp/cb/producer/einou/calendar/pdf/celery.pdf", "11713d5c49b0b8dfefd3e96f626a076b"),
    ("12-001-005", "https://web.archive.org/web/20251224194454if_/https://www.zennoh.or.jp/cb/producer/einou/calendar/pdf/daikon.pdf", "e83548f4937f07bc001d49a3ce920b5d"),
    ("12-001-006", "https://web.archive.org/web/20251219061845if_/https://www.zennoh.or.jp/cb/producer/einou/calendar/pdf/daizu.pdf", "0ef0bf633c65602d62326ca4cee1d224"),
    ("12-001-007", "https://web.archive.org/web/20251216184308if_/https://www.zennoh.or.jp/cb/producer/einou/calendar/pdf/edamame.pdf", "8be22cba4bd966209ff23b30d6c013c4"),
    ("12-001-008", "https://web.archive.org/web/20251224055733if_/https://www.zennoh.or.jp/cb/producer/einou/calendar/pdf/gobou.pdf", "8fd0dcfd838f7f81b26ab5b1381abee5"),
    ("12-001-009", "https://web.archive.org/web/20251220105416if_/https://www.zennoh.or.jp/cb/producer/einou/calendar/pdf/hourensou.pdf", "c7453b23c2ce7fa0f49b02d92f8c547e"),
    ("12-001-010", "https://web.archive.org/web/20251218003416if_/https://www.zennoh.or.jp/cb/producer/einou/calendar/pdf/ichigo.pdf", "9096965004f3a29df6443dfe4c2542fb"),
    ("12-001-011", "https://web.archive.org/web/20251225094557if_/https://www.zennoh.or.jp/cb/producer/einou/calendar/pdf/ingen.pdf", "0ce8de12a9ba555dbeeed1b2bb2d8e5a"),
    ("12-001-012", "https://web.archive.org/web/20260113084236if_/https://www.zennoh.or.jp/cb/producer/einou/calendar/pdf/jagaimo.pdf", "ad5331e96fdaa718f304457a4b5144da"),
    ("12-001-013", "https://web.archive.org/web/20251227205738if_/https://www.zennoh.or.jp/cb/producer/einou/calendar/pdf/kabocha.pdf", "6107fc3a14b1a45102e7f6310c048beb"),
    ("12-001-015", "https://web.archive.org/web/20260109204254if_/https://www.zennoh.or.jp/cb/producer/einou/calendar/pdf/komatsuna.pdf", "0cd6dd1751913bf85942b5abe1575d64"),
    ("12-001-016", "https://web.archive.org/web/20251220080213if_/https://www.zennoh.or.jp/cb/producer/einou/calendar/pdf/kyuri.pdf", "ef3d5bea847c47e1d63e26cf6e359a22"),
    ("12-001-017", "https://web.archive.org/web/20251220101843if_/https://www.zennoh.or.jp/cb/producer/einou/calendar/pdf/leaflettuce.pdf", "1c0ef2359bce03fa6c98bdedfb8ff7ab"),
    ("12-001-019", "https://web.archive.org/web/20260102093401if_/https://www.zennoh.or.jp/cb/producer/einou/calendar/pdf/melon.pdf", "468eb72136de78f941f64e9a304e9e63"),
    ("12-001-020", "https://web.archive.org/web/20260102070509if_/https://www.zennoh.or.jp/cb/producer/einou/calendar/pdf/minitomato.pdf", "4ae0ac534c70e437bd8e6ddf08c22820"),
    ("12-001-021", "https://web.archive.org/web/20251219172212if_/https://www.zennoh.or.jp/cb/producer/einou/calendar/pdf/mizuna.pdf", "87124e82fa137cb76e65050be75db609"),
    ("12-001-022", "https://web.archive.org/web/20251221014948if_/https://www.zennoh.or.jp/cb/producer/einou/calendar/pdf/mugi.pdf", "e82b973fb055d74a5bf48b43180c141c"),
    ("12-001-023", "https://web.archive.org/web/20260102203708if_/https://www.zennoh.or.jp/cb/producer/einou/calendar/pdf/nabana.pdf", "72ad495389cfa6a23d92947d321aa7f5"),
    ("12-001-024", "https://web.archive.org/web/20251218113906if_/https://www.zennoh.or.jp/cb/producer/einou/calendar/pdf/nasu.pdf", "c098308281ed977c10fbb26e84d4cb0d"),
    ("12-001-025", "https://web.archive.org/web/20260128231435if_/https://www.zennoh.or.jp/cb/producer/einou/calendar/pdf/negi.pdf", "a2c6620748dec6622d909e7fac801e72"),
    ("12-001-026", "https://web.archive.org/web/20251218010850if_/https://www.zennoh.or.jp/cb/producer/einou/calendar/pdf/ninjin.pdf", "1115460e47061151c9157c7c1a22e8ba"),
    ("12-001-027", "https://web.archive.org/web/20251218123243if_/https://www.zennoh.or.jp/cb/producer/einou/calendar/pdf/nira.pdf", "12ecb3bb5999058704643a13ce1c0c94"),
    ("12-001-029", "https://web.archive.org/web/20251227100824if_/https://www.zennoh.or.jp/cb/producer/einou/calendar/pdf/pimento.pdf", "520bb4b523a4c0bd6da144a2db575279"),
    ("12-001-030", "https://web.archive.org/web/20251217083613if_/https://www.zennoh.or.jp/cb/producer/einou/calendar/pdf/rakkasei.pdf", "6faf867f8e6585759f8de7563822fb6a"),
    ("12-001-031", "https://web.archive.org/web/20260105201300if_/https://www.zennoh.or.jp/cb/producer/einou/calendar/pdf/satoimo.pdf", "c2384fd7fa401d336302105cf76c41b2"),
    ("12-001-032", "https://web.archive.org/web/20251219075801if_/https://www.zennoh.or.jp/cb/producer/einou/calendar/pdf/satsumaimo.pdf", "b14e079bd5337205804d63866ed2a458"),
    ("12-001-033", "https://web.archive.org/web/20260108232141if_/https://www.zennoh.or.jp/cb/producer/einou/calendar/pdf/shirouri.pdf", "aad24ac06236ff936222071e378209bc"),
    ("12-001-034", "https://web.archive.org/web/20251217021048if_/https://www.zennoh.or.jp/cb/producer/einou/calendar/pdf/shishitou.pdf", "b0dbd946440932a1d6e6bebab18d7f11"),
    ("12-001-035", "https://web.archive.org/web/20251223170848if_/https://www.zennoh.or.jp/cb/producer/einou/calendar/pdf/shouga.pdf", "bba523c48d09ce5d7ac8e5275eb7c67b"),
    ("12-001-038", "https://web.archive.org/web/20251221214709if_/https://www.zennoh.or.jp/cb/producer/einou/calendar/pdf/suika.pdf", "bf19e314bf56452426a6b56dae5352c5"),
    ("12-001-039", "https://web.archive.org/web/20251217135552if_/https://www.zennoh.or.jp/cb/producer/einou/calendar/pdf/tamanegi.pdf", "92afd225ef9d93ec79aa501e2ac6e9dd"),
    ("12-001-040", "https://web.archive.org/web/20251219111003if_/https://www.zennoh.or.jp/cb/producer/einou/calendar/pdf/tomato.pdf", "acc4f57123a597602aff475350be8ff8"),
    ("12-001-041", "https://web.archive.org/web/20251218204053if_/https://www.zennoh.or.jp/cb/producer/einou/calendar/pdf/toumorokoshi.pdf", "0f1088772eead26757ae9097c87d81ea"),
    ("12-001-042", "https://web.archive.org/web/20251228153716if_/https://www.zennoh.or.jp/cb/producer/einou/calendar/pdf/wakenegi.pdf", "9ae219d097c627e32fcdd05b3714017d"),
    ("12-001-043", "https://web.archive.org/web/20251217183416if_/https://www.zennoh.or.jp/cb/producer/einou/calendar/pdf/yamatoimo.pdf", "cf1261f0c91a49899c4b56631e5ad141"),
    ("12-001-044", "https://web.archive.org/web/20251217230507if_/https://www.zennoh.or.jp/cb/producer/einou/calendar/pdf/zucchini.pdf", "502f5a828bba2944e6f506e1169a0b99"),
    ("12-034-001", "https://web.archive.org/web/20240721105416if_/https://www.pref.chiba.lg.jp/ninaite/seikafukyu/documents/r2_02_tubusuke_saibaireki.pdf", "ab7102752b6fd3b4a5f2921efab68862"),
    ("12-035-001", "https://web.archive.org/web/20220805072010if_/http://www.pref.chiba.lg.jp/seisan/documents/kogane.pdf", "d01edca4024453aa626a11c513c37af1"),
    ("12-043-001", "https://web.archive.org/web/20251216113242if_/https://www.zennoh.or.jp/cb/safety/ssi/%E6%B0%B4%E7%A8%B21%E3%81%84%E3%81%99%E3%81%BF%E3%82%B3%E3%82%B7%E3%83%92%E3%82%AB%E3%83%AA.pdf", "5d964c876f3af2bb2a436079b9c1a8b6"),
    ("14-010-001", "https://web.archive.org/web/20240703114159if_/https://ja-kanasei.or.jp/wp-content/uploads/2022/11/saibai_01.pdf", "e3d6dc1d2e3cda51ffad5943dc67447e"),
    ("14-010-002", "https://web.archive.org/web/20240703061608if_/https://ja-kanasei.or.jp/wp-content/uploads/2022/11/saibai_02.pdf", "11bc7c4dcba8b02f2c2d54b12c9e43db"),
    ("14-010-003", "https://web.archive.org/web/20240703164944if_/https://ja-kanasei.or.jp/wp-content/uploads/2022/11/saibai_03.pdf", "365f6cd6a21080fb5e0296df8d3bfa73"),
    ("14-010-004", "https://web.archive.org/web/20240703130417if_/https://ja-kanasei.or.jp/wp-content/uploads/2022/11/saibai_05.pdf", "bf8c381b0cb57d94f2afb2d7e7aca3fa"),
    ("14-010-005", "https://web.archive.org/web/20240705133908if_/https://ja-kanasei.or.jp/wp-content/uploads/2022/11/saibai_06.pdf", "c5c339505a63c1dcc745c3a6ac4452ae"),
    ("14-010-006", "https://web.archive.org/web/20240704000527if_/https://ja-kanasei.or.jp/wp-content/uploads/2022/11/saibai_07.pdf", "34762c1b26dd70cec2781d593dde6d68"),
    ("14-010-007", "https://web.archive.org/web/20240703062628if_/https://ja-kanasei.or.jp/wp-content/uploads/2022/11/saibai_08.pdf", "71de788690d8e1f3e77594af1b584d31"),
    ("14-010-008", "https://web.archive.org/web/20240705222704if_/https://ja-kanasei.or.jp/wp-content/uploads/2022/11/saibai_09.pdf", "b2c7e57fd3298de59079255143a3c785"),
    ("14-010-009", "https://web.archive.org/web/20230328214822if_/https://ja-kanasei.or.jp/wp-content/uploads/2022/11/saibai_10.pdf", "b46fa860864d1fcb30a9faaeb1617437"),
    ("14-010-010", "https://web.archive.org/web/20240703110503if_/https://ja-kanasei.or.jp/wp-content/uploads/2022/11/saibai_11.pdf", "a8af1acfd112b4a0c43e48189adac513"),
    ("14-010-011", "https://web.archive.org/web/20240703063738if_/https://ja-kanasei.or.jp/wp-content/uploads/2022/11/saibai_12.pdf", "d118c34a6e4eda66d338d962aeca71cb"),
    ("14-010-012", "https://web.archive.org/web/20240703101543if_/https://ja-kanasei.or.jp/wp-content/uploads/2022/11/saibai_13.pdf", "6758d352389c5760dae8592365658e69"),
    ("14-010-013", "https://web.archive.org/web/20240701061214if_/https://ja-kanasei.or.jp/wp-content/uploads/2022/11/saibai_14.pdf", "8b5ba1ed1b5fac60a0781dc0b4598a61"),
    ("14-010-014", "https://web.archive.org/web/20240703082853if_/https://ja-kanasei.or.jp/wp-content/uploads/2022/11/saibai_15-1.pdf", "6b3cc9763a612ed022d5fb40cccbf280"),
    ("14-010-015", "https://web.archive.org/web/20240701031423if_/https://ja-kanasei.or.jp/wp-content/uploads/2022/11/saibai_15-2.pdf", "7a9d29322e77b8487b8a652e17da3a92"),
    ("14-010-016", "https://web.archive.org/web/20240703072346if_/https://ja-kanasei.or.jp/wp-content/uploads/2022/11/saibai_16.pdf", "b9a8c0b832329ef2dd05876268ec7d72"),
    ("14-010-017", "https://web.archive.org/web/20240703045500if_/https://ja-kanasei.or.jp/wp-content/uploads/2022/11/saibai_17.pdf", "4b593445a66c37752c96261990c51022"),
    ("14-010-018", "https://web.archive.org/web/20240701022642if_/https://ja-kanasei.or.jp/wp-content/uploads/2022/11/saibai_18.pdf", "05de58df79862bea66215bd86a4514e3"),
    ("14-010-019", "https://web.archive.org/web/20230328205402if_/https://ja-kanasei.or.jp/wp-content/uploads/2022/11/saibai_19.pdf", "95bf9c1699b1f03dffbf86870aa15d90"),
    ("14-010-020", "https://web.archive.org/web/20240704020330if_/https://ja-kanasei.or.jp/wp-content/uploads/2022/11/saibai_20.pdf", "cae4deb1eab54b551f0d294d9a90609d"),
    ("14-010-021", "https://web.archive.org/web/20230328212317if_/https://ja-kanasei.or.jp/wp-content/uploads/2022/11/saibai_21.pdf", "b0da4196c91e660c576351bb0f306b9b"),
    ("14-010-022", "https://web.archive.org/web/20230328204403if_/https://ja-kanasei.or.jp/wp-content/uploads/2022/11/saibai_22.pdf", "860e46dbd45830c54faf00728faf1fe6"),
    ("14-010-023", "https://web.archive.org/web/20240701035258if_/https://ja-kanasei.or.jp/wp-content/uploads/2022/11/saibai_23.pdf", "f921a31c2946033639da2ebde81d9032"),
    ("14-010-024", "https://web.archive.org/web/20240703084712if_/https://ja-kanasei.or.jp/wp-content/uploads/2022/11/saibai_24.pdf", "777d5096418fd6b691ebfb9f95bd0464"),
    ("14-010-025", "https://web.archive.org/web/20240705051644if_/https://ja-kanasei.or.jp/wp-content/uploads/2022/11/saibai_25.pdf", "a7f7dc22cb90615dfb242c01faf4ad8c"),
    ("14-010-026", "https://web.archive.org/web/20240703092140if_/https://ja-kanasei.or.jp/wp-content/uploads/2022/11/saibai_26.pdf", "c76004b82d7d6a2680c1fec4474ae6f7"),
    ("14-010-027", "https://web.archive.org/web/20240701022437if_/https://ja-kanasei.or.jp/wp-content/uploads/2022/11/saibai_27.pdf", "b8a4236e199fa9f3172832e0c076e231"),
    ("14-010-028", "https://web.archive.org/web/20240701070738if_/https://ja-kanasei.or.jp/wp-content/uploads/2022/11/saibai_28.pdf", "91cc1489a25cd1fdd3426397121c60b3"),
    ("14-010-029", "https://web.archive.org/web/20240703151611if_/https://ja-kanasei.or.jp/wp-content/uploads/2022/11/saibai_29.pdf", "7655e12f1a92db7d4d49e8d3e1ff01e9"),
    ("14-010-030", "https://web.archive.org/web/20240703094544if_/https://ja-kanasei.or.jp/wp-content/uploads/2022/11/saibai_30.pdf", "9a038e6b55b0fe2be01f844f668f2681"),
    ("14-010-031", "https://web.archive.org/web/20240703104329if_/https://ja-kanasei.or.jp/wp-content/uploads/2022/11/saibai_31.pdf", "28e958c0c2144339b6d75c3a90cbd569"),
    ("14-010-032", "https://web.archive.org/web/20240703091130if_/https://ja-kanasei.or.jp/wp-content/uploads/2022/11/saibai_32.pdf", "38bb67e4aaa6401dcf97c81c5781a0cf"),
    ("14-010-033", "https://web.archive.org/web/20240703045224if_/https://ja-kanasei.or.jp/wp-content/uploads/2022/11/saibai_33.pdf", "be00814c3dc88a041b5f437393d9fc38"),
    ("14-010-034", "https://web.archive.org/web/20240704144150if_/https://ja-kanasei.or.jp/wp-content/uploads/2022/11/saibai_34.pdf", "2afdb04003ee336921a876df07ef8fe8"),
    ("14-010-035", "https://web.archive.org/web/20230328211623if_/https://ja-kanasei.or.jp/wp-content/uploads/2022/11/saibai_35.pdf", "8bf00eec9a5b9164e8cfad8486575c18"),
    ("14-010-036", "https://web.archive.org/web/20240703090705if_/https://ja-kanasei.or.jp/wp-content/uploads/2022/11/saibai_36.pdf", "4a72d95986dc6c6057c652a1a0e7cbce"),
    ("14-010-037", "https://web.archive.org/web/20240705084227if_/https://ja-kanasei.or.jp/wp-content/uploads/2022/11/saibai_37.pdf", "779470f5c3e06f24fdc05e4b66ec6bd3"),
    ("14-010-038", "https://web.archive.org/web/20240703060448if_/https://ja-kanasei.or.jp/wp-content/uploads/2022/11/saibai_38.pdf", "d90c6e9e5d8e04337d727a3e33abdf13"),
    ("14-010-039", "https://web.archive.org/web/20240701073802if_/https://ja-kanasei.or.jp/wp-content/uploads/2022/11/saibai_39.pdf", "b09b886f6889c27f482ec652f8d07298"),
    ("14-010-040", "https://web.archive.org/web/20240705202455if_/https://ja-kanasei.or.jp/wp-content/uploads/2022/11/saibai_40.pdf", "ef38d6f4d808712d1d9b3b0a9d10647f"),
    ("14-010-041", "https://web.archive.org/web/20240703105335if_/https://ja-kanasei.or.jp/wp-content/uploads/2022/11/saibai_41.pdf", "39184b9861bbcbc4ef317ffb9755a8c4"),
    ("14-010-042", "https://web.archive.org/web/20240703085837if_/https://ja-kanasei.or.jp/wp-content/uploads/2022/11/saibai_42.pdf", "1f30e34b747208910f16ac3cbdd7317c"),
    ("14-010-043", "https://web.archive.org/web/20240703073235if_/https://ja-kanasei.or.jp/wp-content/uploads/2022/11/saibai_43.pdf", "759bfd8dfe576989b5908c79b40194e3"),
    ("14-010-044", "https://web.archive.org/web/20240703212241if_/https://ja-kanasei.or.jp/wp-content/uploads/2022/11/saibai_44.pdf", "116cdcbbc1583880b3e0842b40c73076"),
    ("14-010-045", "https://web.archive.org/web/20240703155932if_/https://ja-kanasei.or.jp/wp-content/uploads/2022/11/saibai_45.pdf", "fa3dd7996f9a397b142b73b8c0ff44bf"),
    ("14-010-046", "https://web.archive.org/web/20240703141038if_/https://ja-kanasei.or.jp/wp-content/uploads/2022/11/saibai_46.pdf", "79511e6b4e29913ab5a225151e680dbc"),
    ("14-010-047", "https://web.archive.org/web/20240703180020if_/https://ja-kanasei.or.jp/wp-content/uploads/2022/11/saibai_47.pdf", "80ce16c736447d39169d79e419aef698"),
    ("14-010-048", "https://web.archive.org/web/20240703122832if_/https://ja-kanasei.or.jp/wp-content/uploads/2022/11/saibai_48.pdf", "aa5ff9ca0fb78506015a2b6438a57e6a"),
    ("14-010-049", "https://web.archive.org/web/20240703051520if_/https://ja-kanasei.or.jp/wp-content/uploads/2022/11/saibai_49.pdf", "978f46a5535b0f9cb2c6d9034724ce3e"),
    ("14-010-050", "https://web.archive.org/web/20240703082808if_/https://ja-kanasei.or.jp/wp-content/uploads/2022/11/saibai_50.pdf", "adf792b4b84b3d0edb9e3a71e0d84fdb"),
    ("14-010-051", "https://web.archive.org/web/20230328213015if_/https://ja-kanasei.or.jp/wp-content/uploads/2022/11/saibai_51.pdf", "8d669a960a9556fb9e1baf9050eb3cb2"),
    ("14-010-052", "https://web.archive.org/web/20240701145059if_/https://ja-kanasei.or.jp/wp-content/uploads/2022/11/saibai_52.pdf", "054178e83f5cc47b3163c7d38b52b59d"),
    ("14-010-053", "https://web.archive.org/web/20240703123229if_/https://ja-kanasei.or.jp/wp-content/uploads/2022/11/saibai_53.pdf", "66cf004163b6afbec84df6c6d4ea0d2d"),
    ("15-051-001", "https://web.archive.org/web/20251216113318if_/https://www.ja-chuetsu.or.jp/hp/wp-content/uploads/2023/06/%E5%A4%A7%E8%B1%86%EF%BC%88%E9%87%8C%E3%81%AE%E3%81%BB%E3%81%BB%E3%81%88%E3%81%BF%EF%BC%89%E6%A0%BD%E5%9F%B9%E6%9A%A6.pdf", "692e0dbe346eac7bbd3ed6b1ce3590fd"),
    ("15-051-002", "https://web.archive.org/web/20251216113323if_/https://www.ja-chuetsu.or.jp/hp/wp-content/uploads/2025/04/R7%E5%A4%A7%E8%B1%86%EF%BC%88%E9%87%8C%E3%81%AE%E3%81%BB%E3%81%BB%E3%81%88%E3%81%BF%EF%BC%89%E6%A0%BD%E5%9F%B9%E6%9A%A6.pdf", "f22b58cde9b9f6f09c0566957437e271"),
    ("15-051-003", "https://web.archive.org/web/20251216134108if_/https://www.ja-chuetsu.or.jp/hp/wp-content/uploads/2023/03/R5.%E3%81%AA%E3%82%93%E3%81%8B%E3%82%93%E7%B1%B3%E3%82%B3%E3%82%B7%E3%83%92%E3%82%AB%E3%83%AA%E6%A0%BD%E5%9F%B9%E6%9A%A6.pdf", "92620b2a1ccdf19ef55bdbfdc0ec590a"),
    ("15-051-004", "https://web.archive.org/web/20251216134119if_/https://www.ja-chuetsu.or.jp/hp/wp-content/uploads/2025/02/%E3%80%90%E3%81%AA%E3%82%93%E3%81%8B%E3%82%93%E7%89%88%E3%80%91R7.%E3%82%B3%E3%82%B7%E3%83%92%E3%82%AB%E3%83%AA%E6%A0%BD%E5%9F%B9%E6%9A%A6.pdf", "8ae2d713d529493287cc182fa13ed463"),
    ("15-051-005", "https://web.archive.org/web/20251216134119if_/https://www.ja-chuetsu.or.jp/hp/wp-content/uploads/2023/06/R5%E6%A0%BD%E5%9F%B9%E6%9A%A6%EF%BC%88%E5%AE%8C%E6%88%90%E7%89%88%EF%BC%89%EF%BC%88%E7%89%B9%E5%88%A5%E6%A0%BD%E5%9F%B9%E7%B1%B3%E3%81%93%E3%81%8C%E3%81%AD%E3%82%82%E3%81%A1%EF%BC%89.pdf", "95c33e48303faaf0bb39daf4cdfe30e5"),
    ("15-051-006", "https://web.archive.org/web/20251216134130if_/https://www.ja-chuetsu.or.jp/hp/wp-content/uploads/2025/04/R7%E5%A4%A7%E8%B1%86%EF%BC%88%E3%82%A8%E3%83%B3%E3%83%AC%E3%82%A4%EF%BC%89%E6%A0%BD%E5%9F%B9%E6%9A%A6.pdf", "9a40ae7ad9e0b37a0bc77498b7b48b2e"),
    ("15-051-007", "https://web.archive.org/web/20251216134130if_/https://www.ja-chuetsu.or.jp/hp/wp-content/uploads/2025/04/R7%E5%A4%A7%E8%B1%86%EF%BC%88%E3%82%A8%E3%83%B3%E3%83%AC%E3%82%A4%EF%BC%89%E6%A0%BD%E5%9F%B9%E6%9A%A6.pdf", "9a40ae7ad9e0b37a0bc77498b7b48b2e"),
    ("15-051-008", "https://web.archive.org/web/20251216134136if_/https://www.ja-chuetsu.or.jp/hp/wp-content/uploads/2023/06/R5%E6%A0%BD%E5%9F%B9%E6%9A%A6%EF%BC%88%E5%AE%8C%E6%88%90%E7%89%88%EF%BC%89%EF%BC%88%E3%81%AB%E3%81%98%E3%81%AE%E3%81%8D%E3%82%89%E3%82%81%E3%81%8D%EF%BC%89.pdf", "ee77c04343a98b82f3a44ec09ab648e5"),
    ("15-051-009", "https://web.archive.org/web/20251216135043if_/https://www.ja-chuetsu.or.jp/hp/wp-content/uploads/2025/04/R7%E3%81%AB%E3%81%98%E3%81%AE%E3%81%8D%E3%82%89%E3%82%81%E3%81%8D%E6%A0%BD%E5%9F%B9%E6%9A%A6.pdf", "d734f2103f27139694add4278533826d"),
    ("15-051-010", "https://web.archive.org/web/20251216135058if_/https://www.ja-chuetsu.or.jp/hp/wp-content/uploads/2024/04/%E2%98%85%E2%91%A3%E4%BB%A4%E5%92%8C6%E5%B9%B4%E5%BA%A6%E3%82%A8%E3%82%B3%E3%83%BB5-5%E6%A0%BD%E5%9F%B9%E6%9A%A6%EF%BC%88%E6%A0%83%E5%B0%BE%E3%83%BB%E5%B1%B1%E5%8F%A4%E5%BF%97%E3%82%B3%E3%82%B7%E3%83%92%E3%82%AB%E3%83%AA%EF%BC%89R6.03.07-.pdf", "ed7e02f2f65117f6f84cb72255ddf89e"),
    ("15-051-011", "https://web.archive.org/web/20251216135114if_/https://www.ja-chuetsu.or.jp/hp/wp-content/uploads/2023/06/R5%E6%A0%BD%E5%9F%B9%E6%9A%A6%EF%BC%88%E5%AE%8C%E6%88%90%E7%89%88%EF%BC%89%EF%BC%88%E4%BA%94%E7%99%BE%E4%B8%87%E7%9F%B3%EF%BC%89.pdf", "4796684fdf08b169b7ee4aa53fe09347"),
    ("15-051-012", "https://web.archive.org/web/20251216135130if_/https://www.ja-chuetsu.or.jp/hp/wp-content/uploads/2023/06/R5%E6%A0%BD%E5%9F%B9%E6%9A%A6%EF%BC%88%E5%AE%8C%E6%88%90%E7%89%88%EF%BC%89%EF%BC%88%E3%82%86%E3%81%8D%E3%82%93%E5%AD%90%E8%88%9E%EF%BC%89.pdf", "db159183311b6dcfac149427b58b0dc5"),
    ("15-051-013", "https://web.archive.org/web/20251216134204if_/https://www.ja-chuetsu.or.jp/hp/wp-content/uploads/2024/05/R6%E3%81%93%E3%81%97%E3%81%84%E3%81%B6%E3%81%8D%E6%A0%BD%E5%9F%B9%E6%9A%A6.pdf", "28d20ba4f0fb443329b1a4586569a40a"),
    ("15-051-014", "https://web.archive.org/web/20251216134210if_/https://www.ja-chuetsu.or.jp/hp/wp-content/uploads/2023/06/R5%E6%A0%BD%E5%9F%B9%E6%9A%A6%EF%BC%88%E5%AE%8C%E6%88%90%E7%89%88%EF%BC%89%EF%BC%88%E6%96%B0%E4%B9%8B%E5%8A%A9%EF%BC%89.pdf", "783ad9e54a43a40d0fe6caffc8230f3a"),
    ("15-051-015", "https://web.archive.org/web/20251216135217if_/https://www.ja-chuetsu.or.jp/hp/wp-content/uploads/2024/04/%E2%98%85%E2%91%A6R6%E3%80%8C%E6%96%B0%E4%B9%8B%E5%8A%A9%E3%80%8D%E6%A0%BD%E5%9F%B9%E6%9A%A6%EF%BC%88%E3%82%A8%E3%82%B3%EF%BC%89R6.03.07.pdf", "c01ef5dc79d99a94004f595db0807f62"),
    ("15-051-016", "https://web.archive.org/web/20251216135233if_/https://www.ja-chuetsu.or.jp/hp/wp-content/uploads/2025/02/%E3%80%90%E3%81%AA%E3%82%93%E3%81%8B%E3%82%93%E7%89%88%E3%80%91R7.%E3%81%93%E3%81%97%E3%81%84%E3%81%B6%E3%81%8D%E6%A0%BD%E5%9F%B9%E6%9A%A6.pdf", "4dd1cc53f3bd5f1dd087f126063f44c4"),
    ("15-051-017", "https://web.archive.org/web/20251216135249if_/https://www.ja-chuetsu.or.jp/hp/wp-content/uploads/2023/06/R5%E6%A0%BD%E5%9F%B9%E6%9A%A6%EF%BC%88%E5%AE%8C%E6%88%90%E7%89%88%EF%BC%89%EF%BC%88%E7%89%B9%E5%88%A5%E6%A0%BD%E5%9F%B9%E7%B1%B3%E3%82%B3%E3%82%B7%E3%83%92%E3%82%AB%E3%83%AA%EF%BC%89.pdf", "98a90ab77abe9dd57d700bc7cdacf7b9"),
    ("15-051-018", "https://web.archive.org/web/20251216135305if_/https://www.ja-chuetsu.or.jp/hp/wp-content/uploads/2023/06/R5%E6%A0%BD%E5%9F%B9%E6%9A%A6%EF%BC%88%E5%AE%8C%E6%88%90%E7%89%88%EF%BC%89%EF%BC%88%E3%82%8F%E3%81%9F%E3%81%BC%E3%81%86%E3%81%97%EF%BC%89.pdf", "7f49d038beba3684507a66fe058d8aac"),
    ("15-051-019", "https://web.archive.org/web/20251216135325if_/https://www.ja-chuetsu.or.jp/hp/wp-content/uploads/2025/04/R7%E3%82%8F%E3%81%9F%E3%81%BC%E3%81%86%E3%81%97%E6%A0%BD%E5%9F%B9%E6%9A%A6.pdf", "255f314526b73bd6569cdc3f9948fee9"),
    ("15-051-020", "https://web.archive.org/web/20251216135337if_/https://www.ja-chuetsu.or.jp/hp/wp-content/uploads/2023/04/R5%E5%A4%A7%E8%B1%86%E6%96%BD%E8%82%A5%E9%98%B2%E9%99%A4%E6%9A%A6.pdf", "d7ce67aedb924a1e1795135a360d1c91"),
    ("15-051-021", "https://web.archive.org/web/20251216135352if_/https://www.ja-chuetsu.or.jp/hp/wp-content/uploads/2024/04/%E2%98%85%E2%91%A5R6%E3%80%8C%E6%96%B0%E4%B9%8B%E5%8A%A9%E3%80%8D%E6%A0%BD%E5%9F%B9%E6%9A%A6%EF%BC%88%E6%85%A3%E8%A1%8C%EF%BC%89R6.03.07.pdf", "1cc4c6ad17ff7c4c47a68ff583eb2f6e"),
    ("15-051-022", "https://web.archive.org/web/20251216135416if_/https://www.ja-chuetsu.or.jp/hp/wp-content/uploads/2024/04/%E2%98%85%E2%91%A0%E4%BB%A4%E5%92%8C6%E5%B9%B4%E5%BA%A6%E3%82%A8%E3%82%B3%E3%83%BB5-5%E6%A0%BD%E5%9F%B9%E6%9A%A6%EF%BC%88%E9%95%B7%E5%B2%A1%E3%82%B3%E3%82%B7%E3%83%92%E3%82%AB%E3%83%AA%EF%BC%89R6.03.07.pdf", "757b9b3f572933639e65b67aa371e155"),
    ("16-013-010", "https://web.archive.org/web/20251226094752if_/https://www.ja-inaba.or.jp/wp/wp-content/uploads/2025/04/f421b988db59ac36201602e8d8ced2cd.pdf", "64642d17f2739bd7211e32b5f88dc230"),
    ("16-062-001", "https://web.archive.org/web/20251216135641if_/https://www.ja-imizuno.or.jp/assets/doc/agric-koyomi/koyomi-suito-r7-01.pdf?20250407", "17ec113cbc87f2ff7f67c1244c8c3056"),
    ("18-079-001", "https://web.archive.org/web/20230330113019if_/https://www.pref.fukui.lg.jp/doc/fukui-noso/shienbu/guri-nnnasaibaihenotennkannsap_d/fil/tokubetusaibaimaininshou.pdf", "3d9fa22ff494aec336a4ac9005a7eb30"),
    ("20-057-001", "https://web.archive.org/web/20251216113433if_/https://www.ja-nagano.iijan.or.jp/Me9wmZUs/wp-content/uploads/2023/06/b8e9e4b1b5a8db5c6e6f41c5bbc79a7a.pdf", "d74f9add10506c4b494eb567d7c26ff2"),
    ("20-058-001", "https://web.archive.org/web/20251216113433if_/https://www.ja-nagano.iijan.or.jp/Me9wmZUs/wp-content/uploads/2025/01/ac893bbc8dbd5fdc9f0ef177c7794d76.pdf", "893e6127bdd14f2f1c52d2685f402f08"),
    ("20-058-002", "https://web.archive.org/web/20231017104708if_/https://www.pref.nagano.lg.jp/nogi/midori/documents/05-2_yamanouchimanual.pdf", "58bb88836f4e1c985afe1a34825c2512"),
    ("20-059-001", "https://web.archive.org/web/20251216113437if_/https://www.pref.nagano.lg.jp/sakuchi/nosei-aec/joho/gijutsu/documents/nakasenanarisaibaisisin.pdf", "ed01ab5a51157966be3dcab32292041c"),
    ("20-060-001", "https://web.archive.org/web/20240527162828if_/https://www.infrc.or.jp/wxp/wp-content/uploads/2016/03/4-2_point_cucumber.pdf", "a291a07356cc27abb3ac2086b3a32bd0"),
    ("21-061-002", "https://web.archive.org/web/20251216113502if_/https://www.zennoh.or.jp/gf/gifu-kome/safety/kijunPdf/r05_cal-akita.pdf", "bb9bf7ed5e77578ee3de2b39911f00c0"),
    ("21-061-004", "https://web.archive.org/web/20251216113506if_/https://www.zennoh.or.jp/gf/gifu-kome/safety/kijunPdf/r05_cal-koshi-hida.pdf", "a30f7f6f5cb57a74ae54232045d02252"),
    ("21-061-006", "https://web.archive.org/web/20251216135633if_/https://www.zennoh.or.jp/gf/gifu-kome/safety/kijunPdf/r05_cal-asahi.pdf", "9184a0e17d5a2ef966436ce918b42bcf"),
]

if __name__ == "__main__":
    main()
