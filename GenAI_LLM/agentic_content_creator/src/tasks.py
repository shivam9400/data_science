from crewai import Task

class ContentTasks:
    def blog_task(self, agent, topic, folder_name):
        return Task(
            description=f"Create a full blog post outline for the topic: {topic}. Focus on a strong hook and 3 key takeaways.",
            expected_output="A structured Markdown blog post outline with a title.",
            agent=agent,
            output_file=f"{folder_name}/1_outline.md"
        )

    def writing_task(self, agent, folder_name):
        return Task(
            description="Write the full blog post based on the outline. Ensure it is at least 600 words and uses Markdown headers.",
            expected_output="A complete Markdown blog post.",
            agent=agent,
            output_file=f"{folder_name}/2_blog_post.md"
        )

    def social_task(self, agent, topic, folder_name):
        return Task(
            description=f"Generate 3 Instagram captions and 1 visual design prompt for {topic}.",
            expected_output="A text file containing the captions and the visual prompt for the designer.",
            agent=agent,
            output_file=f"{folder_name}/3_social_media.md"
        )