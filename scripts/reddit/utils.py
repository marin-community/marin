# Utils specific to reddit processing
import requests

def generate_user(author,i):
    return f"user_{i}"

def fetch_reddit_post(url):
    response = requests.get(f'{url}.json')
    response.raise_for_status()  # Check if the request was successful
    return response.json()

def format_post(post,author_dict):
    output = f"# {post['title']}\n"
    if post['selftext']:
        output += f"\n{post['selftext']}"

    if post['author'] not in author_dict:
        author_dict[post['author']] = generate_user(post['author'],len(author_dict))
    
    output += f" ⏤ by *{author_dict[post['author']]}* (↑ {post['ups']}/ ↓ {post['downs']})"

    return output

def format_comment(comment, escape_new_line, author_dict):
    depth_tag = ''
    
    depth_tag = '\t' * comment['data']['depth']
    depth_tag = f"{depth_tag}- " if depth_tag else '- '

    body = comment['data'].get('body', 'deleted')
    if escape_new_line:
        body = body.replace('\n', '\n' + '\t' * comment['data']['depth'] + '  ')

    if comment['data']['author'] not in author_dict:
        author_dict[comment['data']['author']] = generate_user(comment['data']['author'],len(author_dict))
    
    return f"{depth_tag}{body} ⏤ by *{author_dict[comment['data']['author']]}* (↑ {comment['data']['ups']}/ ↓ {comment['data']['downs']})\n"

def format_comments(comments, escape_new_line, space_comment, author_dict):
    output = ''
    for comment in comments:
        output += format_comment(comment, escape_new_line, author_dict)
        if comment['data'].get('replies'):
            sub_comments = comment['data']['replies']['data']['children']
            output += format_comments(sub_comments, escape_new_line, space_comment, author_dict)
        if comment['data']['depth'] == 0: #and comment['data'].get('replies'):
            if space_comment:
                output += '\n'
    return output

def convert_thread(url, escape_new_line=True, space_comment=True):
    data = fetch_reddit_post(url)
    post = data[0]['data']['children'][0]['data']
    comments = data[1]['data']['children']

    author_dict = {'[deleted]':'[deleted]'}
    output = format_post(post,author_dict)
    output += '\n\n## Comments\n\n'
    output += format_comments(comments, escape_new_line, space_comment, author_dict)

    return output