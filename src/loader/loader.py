import json
import time

import requests


def get_json(url):
    r = requests.get(url)
    if r.ok:
        return r.json()
    print(r.status_code)


def get_total_page(res, page_size=1000):
    return round(res.get('count')/page_size)


def get_url(page, page_size=1000, ch='246_244'):
    return 'https://dom.ria.com/node/searchEngine/v2/?category=1&realty_type=2&operation=3&state_id=0&city_id=0&in_radius=0&with_newbuilds=0&price_cur=1&wo_dupl=1&complex_inspected=0&sort=inspected_sort&period=0&notFirstFloor=0&notLastFloor=0&with_map=0&photos_count_from=0&firstIteraction=false&fromAmp=0&type=search&client=searchV2&limit={}&page={}&operation_type=3&ch={}'.format(page_size, page, ch)


def load_ria():
    total_pages = get_total_page(get_json(get_url(0)))
    print(total_pages)
    items_ids = []
    chs = ['235_f_0%2C235_t_9000%2C246_244',
           '235_f_9001%2C235_t_12000%2C246_244',
           '235_f_12001%2C235_t_15000%2C246_244',
           '235_f_15001%2C235_t_20000%2C246_244',
           '235_f_20001%2C235_t_28000%2C246_244',
           '235_f_28001%2C235_t_37000%2C246_244',
           '235_f_37001%2C235_t_48000%2C246_244',
           '235_f_48001%2C235_t_70000%2C246_244',
           '235_f_70001%2C235_t_7000000%2C246_244']
    for ch in chs:
        total_pages = get_total_page(get_json(get_url(0, ch=ch)))
        for page in range(0, total_pages):
            res = get_json(get_url(page, ch=ch))
            items_ids.extend(res.get('items'))

    unique_item_ids = set(items_ids)
    print(len(unique_item_ids))

    items = []

    for _id in unique_item_ids:
        items.append(get_json('https://dom.ria.com/node/searchEngine/v2/view/realty/{}?lang_id=4'.format(_id)))
        if len(items) % 1000 == 0:
            print(len(items))

    with open("items_ria.json", "w") as outfile:
        json.dump(items, outfile)


def load_lun_2():
    items = []
    file = "items_lun.json"
    section_id = 2
    no_update = 0
    for price_range in range(0, 10000):
        print('price_range {}'.format(price_range))
        no_update = no_update + 1

        for page in range(1, 100):
            page_items = get_json(
                "https://flatfy.ua/api/realties?lang=uk&price_min={}&price_max={}&section_id={}&currency=UAH&page={}".format(
                    price_range * 100 + 1, (price_range + 1) * 100, section_id, page)).get('data')
            if len(page_items) == 0:
                break
            no_update = 0
            items.extend(page_items)
            print(len(items))

        if no_update > 100:
            break

    with open(file, "w") as outfile:
        json.dump(items, outfile)


def load_lun():
    items = []
    file = "items_lun_1.json"
    section_id = 1
    no_update = 0
    for price_range in range(9430, 100000):
        print('price_range {}'.format(price_range))
        no_update = no_update + 1

        for page in range(1, 100):
            tries = 100
            page_items = []
            while tries > 0:
                try:
                    page_items = get_json(
                        "https://flatfy.ua/api/realties?lang=uk&price_min={}&price_max={}&section_id={}&currency=USD&page={}".format(
                            price_range * 100 + 1, (price_range + 1) * 100, section_id, page)).get('data')
                    break
                except:
                    print('Error')
                    tries = tries - 1
                    time.sleep(5)

            if len(page_items) == 0:
                break
            no_update = 0
            items.extend(page_items)
            print(len(items))

        if no_update > 1000:
            break

    with open(file, "w") as outfile:
        json.dump(items, outfile)


if __name__ == '__main__':
    load_lun()