# get current position of y scrollbar
last_position = driver.execute_script("return window.pageYOffset;")

reviews = []
review_ids = set()

while True:
    # get cards on the page
    cards = driver.find_elements_by_class_name('apphub_Card')

    for card in cards[-20:]:  # only the tail end are new cards
        # gamer profile url
        profile_url = card.find_element_by_xpath('.//div[@class="apphub_friend_block"]/div/a[2]').get_attribute('href')

        # steam id
        steam_id = profile_url.split('/')[-2]
        
        # check to see if I've already collected this review
        if steam_id in review_ids:
            continue
        else:
            review_ids.add(steam_id)

        # username
        user_name = card.find_element_by_xpath('.//div[@class="apphub_friend_block"]/div/a[2]').text

        # language of the review
        date_posted = card.find_element_by_xpath('.//div[@class="apphub_CardTextContent"]/div').text
        review_content = card.find_element_by_xpath('.//div[@class="apphub_CardTextContent"]').text.replace(date_posted,'').strip()      

        # recommendation
        thumb_text = card.find_element_by_xpath('.//div[@class="reviewInfo"]/div[2]').text

        # save review
        review = (steam_id, review_content, thumb_text, date_posted)
        reviews.append(review)

    # Scroll down
    driver.find_element_by_tag_name('body').send_keys(Keys.PAGE_DOWN)
    sleep(2)  # Wait for a moment to let the content load
    
    # get current position after scrolling
    curr_position = driver.execute_script("return window.pageYOffset;")

    # Break the loop if no new content is loaded
    if curr_position == last_position:
        break

    last_position = curr_position

# Close the WebDriver when done
driver.quit()