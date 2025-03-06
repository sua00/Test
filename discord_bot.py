import discord
import os
from discord.ext import commands
from dotenv import load_dotenv
from harry_potter_rag import search_similar_scenes, generate_response

load_dotenv()

TOKEN = os.getenv('DISCORD_BOT_TOKEN')

intents = discord.Intents.default()
intents.messages = True
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

# 기본 캐릭터와 역할 설정 (유저별로 다르게 관리도 가능)
character_choices = ["Harry", "Ron", "Hermione"]
role_choices = ["friend", "professor", "family"]
current_character = "Harry"
current_role = "friend"

@bot.event
async def on_ready():
    print(f'✅ Bot is ready! Logged in as {bot.user}')

@bot.command(name="set_character")
async def set_character(ctx, character: str):
    global current_character
    if character in character_choices:
        current_character = character
        await ctx.send(f"🪄 Character set to **{character}**!")
    else:
        await ctx.send(f"❌ Invalid character. Choose from: {', '.join(character_choices)}")

@bot.command(name="set_role")
async def set_role(ctx, role: str):
    global current_role
    if role in role_choices:
        current_role = role
        await ctx.send(f"🎭 Role set to **{role}**!")
    else:
        await ctx.send(f"❌ Invalid role. Choose from: {', '.join(role_choices)}")

@bot.command(name="ask")
async def ask_question(ctx, *, question: str):
    await ctx.send(f"🔍 Searching for scenes and generating response from **{current_character}**...")

    # 장면 검색
    retrieved_scenes = search_similar_scenes(question)

    # Gemini 응답 생성
    response = generate_response(
        character=current_character,
        role=current_role,
        question=question,
        retrieved_scenes="\n\n".join(retrieved_scenes)
    )

    await ctx.send(f"**[{current_character} to {current_role}]**: {response}")

bot.run(TOKEN)
