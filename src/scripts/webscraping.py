from src.utils.utils import pd, Options, webdriver, WebDriverWait, ec, By, time, Keys, re
from src.utils.utils import s_exceptions


pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)


def set_browser_avoid_bot_detection():
    """
    Trying to bypass website bot detection
    :return: Chrome Driver
    """

    options = Options()
    # Adding argument to disable the 'auto-close' browser
    options.add_experimental_option("detach", True)
    # Adding argument to disable the AutomationControlled flag
    options.add_argument("--disable-blink-features=AutomationControlled")
    # Exclude the collection of enable-automation switches
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    # Turn-off userAutomationExtension
    options.add_experimental_option("useAutomationExtension", False)
    pre_driver = webdriver.Chrome(options)

    return pre_driver


def prepare_page(driver):
    """
    Prepare the website, removing cookie panels, and other automatic elements
    :param driver: Chrome Driver
    """

    cookie_button = WebDriverWait(driver, 20).until(ec.element_to_be_clickable((By.ID, 'didomi-notice-agree-button')))
    cookie_button.click()
    time.sleep(5)
    bot_button = WebDriverWait(driver, 20).until(ec.element_to_be_clickable((By.XPATH, '/html/body/div[2]/div[3]/div[1]')))
    bot_button.click()
    driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")
    time.sleep(5)
    close_bot_button = WebDriverWait(driver, 20).until(ec.element_to_be_clickable((By.XPATH, '/html/body/div[2]/div[2]/div/div')))
    close_bot_button.click()
    driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.CONTROL + Keys.HOME)
    time.sleep(5)


def check_elements(list_of_equipment, house_elements):
    """
    Check what elements or parameters each house has
    :param list_of_equipment: All the elements that a house can have
    :param house_elements: The elements that each house has
    :return: Checked elements, two values (0, 1)
    """

    air_conditioning, wardrobes, elevator, heating, garage, terrace = int(), int(), int(), int(), int(), int()

    for m in range(len(list_of_equipment)):
        if list_of_equipment[m] in house_elements:
            if m == 0:
                air_conditioning = 1
            elif m == 1:
                wardrobes = 1
            elif m == 2:
                elevator = 1
            elif m == 3:
                heating = 1
            elif m == 4:
                garage = 1
            elif m == 5:
                terrace = 1
        else:
            if m == 0:
                air_conditioning = 0
            elif m == 1:
                wardrobes = 0
            elif m == 2:
                elevator = 0
            elif m == 3:
                heating = 0
            elif m == 4:
                garage = 0
            elif m == 5:
                terrace = 0

    return air_conditioning, wardrobes, elevator, heating, garage, terrace


def get_features(driver, madrid_location, place):
    """
    Take all the characteristics of each house
    :param driver: Chrome Driver
    :param madrid_location: Location of the house
    :param place: Type of house
    :return: Dictionary with the house parameters
    """

    price = WebDriverWait(driver, 5).until(ec.presence_of_element_located((By.XPATH, '/html/body/div[2]/div[2]/section[2]/main/section[1]/div[1]/div[1]/span'))).text
    price = re.split(" ", price)[0].replace(".", "")
    rooms = WebDriverWait(driver, 5).until(ec.presence_of_element_located((By.CLASS_NAME, 'icon-room'))).text
    toilets = WebDriverWait(driver, 5).until(ec.presence_of_element_located((By.CLASS_NAME, 'icon-bath'))).text
    area = WebDriverWait(driver, 5).until(ec.presence_of_element_located((By.CLASS_NAME, 'icon-meter'))).text
    area = re.split(" ", area)[0].replace(".", "")
    WebDriverWait(driver, 5).until(ec.presence_of_element_located((By.XPATH, '/html/body/div[2]/div[2]/section[2]/main/section[2]/h3')))
    equipment_elements = WebDriverWait(driver, 5).until(ec.presence_of_all_elements_located((By.CSS_SELECTOR, 'li.hIcon.feature')))
    elements = [element.text for element in equipment_elements]
    air_conditioning, wardrobes, elevator, heating, garage, terrace = check_elements(elements_list, elements)
    data = \
        {
            "Place": place,
            "Location": madrid_location,
            "Rooms": rooms,
            "Toilets": toilets,
            "Area": area,
            "Air Conditioning": air_conditioning,
            "Built-in Wardrobes": wardrobes,
            "Elevator": elevator,
            "Heating": heating,
            "Garage": garage,
            "Terrace": terrace,
            "Price": price
        }

    return data


def get_home_info(driver, madrid_location):
    """
    Collect all the information about each house
    :param driver: Chrome Driver
    :param madrid_location: House location
    :return: Dataframe with all houses information of each location
    """

    flag = True
    dataframe_list = []

    while flag:
        data_list = []
        actual_url = driver.current_url
        places = WebDriverWait(driver, 5).until(ec.presence_of_all_elements_located((By.XPATH, '//h3[@class="title logo-aside"]/a')))
        houses_links = [house.get_attribute('href') for house in places]
        time.sleep(1)
        place_type = [re.split(" ", place.text)[0] for place in places]

        for n in range(len(houses_links)):

            try:
                time.sleep(1)
                driver.get(houses_links[n])
            except s_exceptions.ElementClickInterceptedException:
                WebDriverWait(google_driver, 10).until(ec.element_to_be_clickable((By.XPATH, '/html/body/div[6]/div/div/div[1]/div'))).click()
                time.sleep(1.5)
                driver.get(houses_links[n])

            time.sleep(1)

            try:
                data = get_features(driver, madrid_location, place_type[n])
                data_list.append(data)
            except s_exceptions.TimeoutException:
                continue

        page_df = pd.DataFrame(data_list)
        dataframe_list.append(page_df)

        try:
            driver.get(actual_url)
            time.sleep(2)
            driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")
            time.sleep(5)
            WebDriverWait(driver, 5).until(ec.element_to_be_clickable((By.CSS_SELECTOR, 'button.button.next-button'))).click()
            flag = True
            time.sleep(1)
        except s_exceptions.TimeoutException:
            flag = False

    zone_df = pd.concat(dataframe_list, ignore_index=True)

    return zone_df


def get_zone(driver):
    """
    Coger todas las zonas que nos interesan
    :param driver: Chrome Driver
    :return: List of all locations and their links
    """

    WebDriverWait(driver, 5).until(ec.presence_of_element_located((By.XPATH, '/html/body/div[2]/div[1]/div[2]/div/ul/li[3]/div/div'))).click()
    madrid_zones = WebDriverWait(driver, 5).until(ec.presence_of_all_elements_located((By.XPATH, '//li[@class="flex flex-nowrap"]/a')))
    zones = [zone.text for zone in madrid_zones]
    links = [house.get_attribute('href') for house in madrid_zones]

    return links, zones


if __name__ == '__main__':

    elements_list = ["Aire acondicionado", "Armarios empotrados", "Ascensor", "Calefacción", "Garaje", "Terraza"]

    google_driver = set_browser_avoid_bot_detection()
    google_driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    houses_page_sale = "https://www.yaencontre.com/venta/pisos/madrid"
    google_driver.get(houses_page_sale)
    prepare_page(google_driver)
    zone_links, locations = get_zone(google_driver)
    complete_df_list = []
    for i in range(len(zone_links)):
        google_driver.get(zone_links[i])
        zone_dataframe = get_home_info(google_driver, locations[i])
        complete_df_list.append(zone_dataframe)

    final_dataframe = pd.concat(complete_df_list, ignore_index=True)
    final_dataframe.to_csv('./data/raw/houses_info.csv', index=False)
