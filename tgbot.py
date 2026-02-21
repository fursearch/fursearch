"""Telegram bot for fursuit character identification using SAM3 system."""

import asyncio
import html
import os
import random
import re
import sys
import time
import traceback
from pathlib import Path
from tempfile import NamedTemporaryFile

from aiohttp import web
from PIL import Image
from telegram import ReactionTypeEmoji, Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

import aitool

start_time = time.perf_counter()

from sam3_fursearch import FursuitIdentifier, FursuitIngestor, Config
from sam3_fursearch.api.identifier import discover_datasets, merge_multi_dataset_results, merge_with_preferred
from sam3_fursearch.api.annotator import annotate_image
from telegram import InputMediaPhoto

from sam3_fursearch.storage.database import Database, get_git_version, get_source_url, get_source_image_url
from sam3_fursearch.storage.database import SOURCE_TGBOT
from sam3_fursearch.storage.tracking import IdentificationTracker
from typing import Optional

# Pattern to match "character:Name" in caption
CHARACTER_PATTERN = re.compile(r"character:(\S+)", re.IGNORECASE)
# Pattern to match "/fursearch CharName" in caption
FURSEARCH_CAPTION_PATTERN = re.compile(r"^/fursearch\s+(.+)", re.IGNORECASE)

# Global instances (lazy loaded)
_identifiers: Optional[list[FursuitIdentifier]] = None
_ingestor: Optional[FursuitIngestor] = None
_tracker: Optional[IdentificationTracker] = None

# Media group buffering: media_group_id -> list of Update objects
_media_group_buffers: dict[str, list[Update]] = {}
_media_group_tasks: dict[str, asyncio.Task] = {}
MEDIA_GROUP_TIMEOUT = 1.0  # seconds to wait for more photos in a group


def _get_submission_character_url(post_id: str) -> Optional[str]:
    """Look up character_url from submission_metadata across all datasets."""
    try:
        for ident in get_identifiers():
            if Path(ident.db.db_path).stem != Config.DEFAULT_DATASET:
                continue
            meta = ident.db.get_submission_metadata(post_id)
            if meta and meta.get("character_url"):
                return meta["character_url"]
    except Exception:
        pass
    return None


def _save_submission_metadata(post_id: str, character_url: Optional[str] = None,
                              submitted_by: Optional[str] = None):
    """Persist submission metadata to the default dataset DB."""
    try:
        ingestor = get_ingestor()
        ingestor.db.set_submission_metadata(post_id, character_url=character_url, submitted_by=submitted_by)
    except Exception as e:
        print(f"Warning: could not persist submission metadata: {e}", file=sys.stderr)


def _get_page_url(source: Optional[str], post_id: str, character_name: Optional[str] = None) -> Optional[str]:
    """Get a page URL for a detection."""
    if source == SOURCE_TGBOT:
        return _get_submission_character_url(post_id)
    return get_source_url(source, post_id, character_name=character_name)


def parse_character_input(text: str) -> tuple[str, Optional[str]]:
    """Parse a character name input, stripping prefixes and extracting @username.

    Returns (character_name, character_url). If @username is present, character_url
    is set to the t.me link. If the name is empty after stripping, the username is
    used as the character name.
    """
    text = re.sub(r"^this\s+is\s+", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^character:", "", text, flags=re.IGNORECASE).strip()

    character_url = None
    at_match = re.search(r"@(\w+)", text)
    if at_match:
        character_url = f"https://t.me/{at_match.group(1)}"
        text = text.replace(at_match.group(0), "").strip().lower()
        if not text:
            text = at_match.group(1)
    return text.lower(), character_url



def get_identifiers():
    global _identifiers
    if _identifiers:
        return _identifiers
    base_dir = Config.BASE_DIR
    datasets = discover_datasets(base_dir)
    if not datasets:
        raise FileNotFoundError(
            f"No datasets found in {base_dir}. "
            "Expected *.db and *.index file pairs."
        )
    allowed_sources_env = os.environ.get("ALLOWED_SOURCES", "").strip()
    if allowed_sources_env:
        print(f'Allowed sources: {allowed_sources_env}')
        allowed = [s.strip() for s in allowed_sources_env.split(",") if s.strip()]
        datasets = [(db, idx) for db, idx in datasets if Path(db).stem in allowed]

    names = [Path(db_path).stem for db_path, _ in datasets]
    print(f"Auto-discovered {len(names)} dataset(s): {', '.join(names)}")
    _identifiers = [
        FursuitIdentifier(
            db_path=db_path,
            index_path=index_path,
            segmentor_model_name=Config.SAM3_MODEL,
            segmentor_concept=Config.DEFAULT_CONCEPT,
        )
        for db_path, index_path in datasets
    ]
    return _identifiers


def get_ingestor() -> FursuitIngestor:
    """Get or create the ingestor instance (writes to DB)."""
    global _ingestor
    if _ingestor is None:
        _ingestor = FursuitIngestor(segmentor_model_name=Config.SAM3_MODEL, segmentor_concept=Config.DEFAULT_CONCEPT)
    return _ingestor


def invalidate_identifiers():
    """Clear cached identifiers so they reload from disk on next use."""
    global _identifiers
    _identifiers = None


def get_tracker() -> IdentificationTracker:
    """Get or create the identification tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = IdentificationTracker()
    return _tracker


async def photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle photo messages.

    If the photo is part of a media group (multiple photos sent at once),
    buffers all photos and processes them together after a short delay.
    Otherwise processes the single photo immediately.
    """
    if not update.message:
        print("Invalid message", file=sys.stderr)
        return

    attachment = update.message.effective_attachment
    if not attachment:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Please send a photo."
        )
        return

    media_group_id = update.message.media_group_id
    if media_group_id:
        # Buffer this photo and schedule grouped processing
        if media_group_id not in _media_group_buffers:
            _media_group_buffers[media_group_id] = []
        _media_group_buffers[media_group_id].append(update)
        print(f"Buffered photo for media group {media_group_id} ({len(_media_group_buffers[media_group_id])} so far)")

        # Cancel any existing timer and start a new one
        if media_group_id in _media_group_tasks:
            _media_group_tasks[media_group_id].cancel()
        _media_group_tasks[media_group_id] = asyncio.create_task(
            _process_media_group_after_delay(media_group_id, context)
        )
        return

    # Single photo - process immediately
    await _handle_single_photo(update, context)


async def _process_media_group_after_delay(media_group_id: str, context: ContextTypes.DEFAULT_TYPE):
    """Wait for all photos in a media group to arrive, then process them."""
    await asyncio.sleep(MEDIA_GROUP_TIMEOUT)

    updates = _media_group_buffers.pop(media_group_id, [])
    _media_group_tasks.pop(media_group_id, None)

    if not updates:
        return

    # Sort by message_id to preserve order
    updates.sort(key=lambda u: u.message.message_id)

    # Caption is typically only on the first message in a media group
    caption = ""
    for u in updates:
        if u.message.caption:
            caption = u.message.caption
            break

    print(f"Processing media group {media_group_id}: {len(updates)} photo(s), caption: {caption}")

    # Check if this is an add request
    fursearch_match = FURSEARCH_CAPTION_PATTERN.search(caption)
    char_match = CHARACTER_PATTERN.search(caption)

    chat_id = updates[0].effective_chat.id
    first_msg = updates[0].message
    total = len(updates)

    if fursearch_match:
        character_name = fursearch_match.group(1).strip()
        if character_name:
            await _add_photo_group(updates, context, character_name, chat_id, first_msg, total)
            return

    if char_match:
        character_name = char_match.group(1)
        await _add_photo_group(updates, context, character_name, chat_id, first_msg, total)
        return

    # Identify all photos in the group
    await _react(first_msg, "👀")
    status_msg = await context.bot.send_message(
        chat_id=chat_id, reply_to_message_id=first_msg.message_id,
        text=f"Identifying {total} photo(s)...")

    results = []  # list of (annotated_path, caption_html)
    errors = 0
    for i, u in enumerate(updates):
        await _edit_status(status_msg, f"Identifying {total} photo(s)... ({i+1}/{total})")
        try:
            bot_me = await context.bot.get_me()
            annotated_path, caption_html, error = await _identify_photo_to_result(
                u.message.effective_attachment, chat_id,
                user=u.effective_user, message_id=u.message.message_id,
                bot_username=bot_me.username,
            )
            if error:
                errors += 1
            else:
                results.append((annotated_path, caption_html))
        except Exception:
            traceback.print_exc()
            errors += 1

    await _delete_message(status_msg)

    reply_kwargs = {"chat_id": chat_id, "reply_to_message_id": first_msg.message_id}

    # Try sending as a media group if all captions fit within Telegram's 1024-char limit
    sent_as_group = False
    if len(results) >= 2 and all(len(cap) <= 1024 for _, cap in results):
        media = [
            InputMediaPhoto(media=open(path, 'rb'), caption=cap, parse_mode="HTML")
            for path, cap in results
        ]
        try:
            await context.bot.send_media_group(**reply_kwargs, media=media)
            sent_as_group = True
        except Exception:
            traceback.print_exc()

    # Fall back to separate photo + text messages
    if not sent_as_group:
        for path, cap in results:
            try:
                with open(path, 'rb') as photo_file:
                    await context.bot.send_photo(**reply_kwargs, photo=photo_file)
                await context.bot.send_message(
                    **reply_kwargs, text=cap, parse_mode="HTML",
                    disable_web_page_preview=True)
            except Exception:
                traceback.print_exc()
                errors += 1

    for path, _ in results:
        try:
            os.unlink(path)
        except OSError:
            pass

    if errors:
        await context.bot.send_message(
            **reply_kwargs, text=f"Failed to identify {errors}/{total} photo(s).")


async def _add_photo_group(updates: list[Update], context: ContextTypes.DEFAULT_TYPE,
                           character_name: str, chat_id: int, first_msg, total: int):
    """Add multiple photos for a character from a media group, with a single summary."""
    await _react(first_msg, "🗿")
    status_msg = await context.bot.send_message(
        chat_id=chat_id, reply_to_message_id=first_msg.message_id,
        text=f"Adding {total} photo(s) for '{character_name}'...")
    added_total = 0
    for i, u in enumerate(updates):
        added_total += await add_photo(u, context, character_name, silent=True)
        await _edit_status(status_msg, f"Adding {total} photo(s) for '{character_name}'... ({i+1}/{total})")
    await _delete_message(status_msg)
    invalidate_identifiers()
    if added_total > 0:
        await context.bot.send_message(
            chat_id=chat_id, reply_to_message_id=first_msg.message_id,
            text=f"Added {total} photo(s) for '{character_name}'.")
    else:
        await context.bot.send_message(
            chat_id=chat_id, reply_to_message_id=first_msg.message_id,
            text=f"Failed to add any images for '{character_name}'. No segments detected.")


async def _handle_single_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Process a single (non-grouped) photo message."""
    caption = update.message.caption or ""
    print(f"Received photo with caption: {caption}")

    # Check if this is an add request via /fursearch caption
    fursearch_match = FURSEARCH_CAPTION_PATTERN.search(caption)
    if fursearch_match:
        character_name = fursearch_match.group(1).strip()
        if character_name:
            await add_photo(update, context, character_name)
            return

    # Check if this is an add request via character:Name caption
    match = CHARACTER_PATTERN.search(caption)
    if match:
        await add_photo(update, context, match.group(1))
    else:
        await identify_photo(update, context)

async def download_tg_file(new_file):
    tg_dir = f"datasets/{Config.DEFAULT_DATASET}/tg_download"
    os.makedirs(tg_dir, exist_ok=True)
    temp_path = Path(tg_dir) / f"{new_file.file_id}.jpg"
    print(f"Downloading into {temp_path}")
    with open(temp_path, 'wb') as f:
        bs = await new_file.download_as_bytearray()
        f.write(bs)
        f.flush()
    return temp_path

def make_tgbot_post_id(chat_id: int, msg_id: int, file_id: str) -> str:
    """Create a unique post_id from telegram message identifiers."""
    return f"{chat_id}_{msg_id}_{file_id}"


async def _react(message, emoji: str):
    """Set a reaction on a message. Best-effort, silently ignores errors."""
    try:
        await message.set_reaction([ReactionTypeEmoji(emoji=emoji)])
    except Exception:
        pass


async def _edit_status(message, text: str):
    """Edit a status message. Best-effort, silently ignores errors (e.g. text unchanged)."""
    try:
        await message.edit_text(text)
    except Exception:
        pass


async def _delete_message(message):
    """Delete a message. Best-effort, silently ignores errors."""
    try:
        await message.delete()
    except Exception:
        pass


def _get_sender_url(user) -> Optional[str]:
    """Get a t.me/ URL for a Telegram user, or user ID as fallback."""
    if user and user.username:
        return f"https://t.me/{user.username.lower()}"
    if user:
        return str(user.id)
    return None


async def add_photo(update: Update, context: ContextTypes.DEFAULT_TYPE, character_name: str,
                    photo_message=None, silent: bool = False) -> int:
    """Add a photo to the database for a character.

    Args:
        photo_message: If provided, use this message's photo instead of update.message.
        silent: If True, don't send per-photo messages or reactions (for batch use).

    Returns:
        Number of images added (0 on failure).
    """
    msg = photo_message or update.message
    if not silent:
        await _react(update.message, "🗿")
    attachment = msg.effective_attachment
    new_file = await attachment[-1].get_file()
    user = update.effective_user

    character_name, character_url = parse_character_input(character_name)
    uploaded_by = _get_sender_url(user)

    if character_url and character_url.lower() != (uploaded_by or "").lower():
        if not silent:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="For now you can only tag your own telegram username. Photos have been submitted with a character name instead. Please come back later as this is being implemented."
            )
        character_url = None

    post_id = make_tgbot_post_id(update.effective_chat.id, msg.message_id, new_file.file_id)
    try:
        temp_path = await download_tg_file(new_file)
        # Rename temp file to use post_id so identifier extracts it correctly
        post_id_path = temp_path.parent / f"{post_id}.jpg"
        temp_path.rename(post_id_path)
        ingestor = get_ingestor()
        added = await asyncio.to_thread(
            ingestor.add_images,
            character_names=[character_name],
            image_paths=[str(post_id_path)],
            source=SOURCE_TGBOT,
            uploaded_by=uploaded_by,
            add_full_image=True,
        )

        # Clean up temp file
        # os.unlink(temp_path)

        if added > 0:
            if not silent:
                invalidate_identifiers()
            _save_submission_metadata(post_id, character_url=character_url, submitted_by=uploaded_by)
            if not silent:
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text=f"Added image for character '{character_name}'."
                )
            return added
        else:
            if not silent:
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text=f"Failed to add image for '{character_name}'. No segments detected."
                )
            return 0

    except Exception as e:
        traceback.print_exc()
        if not silent:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="Error adding image. Please try again."
            )
        return 0


async def _identify_photo_to_result(photo_attachment, chat_id: int,
                                    user=None, message_id: int = None,
                                    bot_username: str = None):
    """Download a photo, run identification, and return the annotated image path + caption.

    Returns (annotated_path, caption_html, error_text). On failure, annotated_path and
    caption_html are None and error_text describes the problem.
    """
    new_file = await photo_attachment[-1].get_file()
    temp_path = await download_tg_file(new_file)
    image = Image.open(temp_path)

    identifiers = get_identifiers()

    def _run_identify():
        print(f"Running identification on {temp_path} with {len(identifiers)} dataset(s)...")
        all_results = [ident.identify(image, top_k=Config.DEFAULT_TOP_K) for ident in identifiers]
        dataset_names = [Path(ident.db.db_path).stem for ident in identifiers]
        for ident, ds_results in zip(identifiers, all_results):
            for res in ds_results:
                print(f"Segment {res.segment_index}. ({res.segment_confidence:.2f}): "
                      f"{len(res.matches)} matches:")
                for m in res.matches:
                    print(f"{Path(ident.db.db_path).stem}: {m.character_name} ({m.confidence:.2f})")
        preferred_env = os.environ.get("PREFERRED_DATASETS", Config.PREFERRED_DATASETS)
        preferred_names = {s.strip() for s in preferred_env.split(",") if s.strip()}
        num_preferred = int(os.environ.get("NUM_PREFERRED_RESULTS", Config.NUM_PREFERRED_RESULTS))
        merged = merge_with_preferred(
            all_results, dataset_names, preferred_names, num_preferred,
            top_k=Config.TGBOT_MAX_RESULTS,
        )
        return all_results, merged

    t0 = time.monotonic()
    all_results, results = await asyncio.to_thread(_run_identify)
    processing_time_ms = int((time.monotonic() - t0) * 1000)

    if not results:
        _log_tracking(identifiers, all_results, results, [], image, str(temp_path),
                      processing_time_ms, chat_id, message_id, user)
        return None, None, "No matching characters found."

    min_confidence = Config.DEFAULT_MIN_CONFIDENCE
    lines = []
    shown_matches = []  # (seg_idx, match, rank_after_merge)
    for i, result in enumerate(results, 1):
        filtered = [m for m in result.matches if m.confidence >= min_confidence]
        if not filtered:
            continue
        lines.append(f"Character {i}:")
        for n, m in enumerate(filtered):
            url = _get_page_url(m.source, m.post_id, m.character_name)
            # img_url = get_source_image_url(m.source, m.post_id)
            name = html.escape(m.character_name or 'Unknown')
            # pct = f"{m.confidence*100:.1f}%"
            name_part = f"<a href=\"{url}\">{name}</a>" if url else name
            # pct_part = f"<a href=\"{img_url}\">{pct}</a>" if img_url else pct
            lines.append(f"  {n+1}. {name_part}")
            shown_matches.append((i - 1, m, n))
        lines.append(f"")

    _log_tracking(identifiers, all_results, results, shown_matches, image,
                  str(temp_path), processing_time_ms, chat_id, message_id, user)

    if not lines:
        return None, None, f"No matches above {min_confidence:.0%} confidence."

    # Resize image to match pipeline input size so bounding boxes align
    if max(*image.size) > Config.MAX_INPUT_IMAGE_SIZE:
        w, h = image.size
        if w >= h:
            new_w, new_h = Config.MAX_INPUT_IMAGE_SIZE, int(h * Config.MAX_INPUT_IMAGE_SIZE / w)
        else:
            new_h, new_w = Config.MAX_INPUT_IMAGE_SIZE, int(w * Config.MAX_INPUT_IMAGE_SIZE / h)
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    watermark_handle = f"@{bot_username}" if bot_username else "@FurSearchBot"
    watermark_text = f"{watermark_handle} {get_git_version()}"
    annotated = await asyncio.to_thread(annotate_image, image, results, min_confidence, watermark_text)
    with NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        annotated.save(f, format="JPEG", quality=90)
        temp_annotated_path = f.name

    caption_html = "\n".join(lines)
    print(caption_html)
    return temp_annotated_path, caption_html, None


async def identify_and_send(context: ContextTypes.DEFAULT_TYPE, chat_id: int,
                           photo_attachment, reply_to_message_id: int = None,
                           react_message=None, user=None, message_id: int = None):
    """Download photo, identify characters, and send annotated result."""
    if react_message:
        await _react(react_message, "👀")

    reply_kwargs = {"chat_id": chat_id}
    if reply_to_message_id:
        reply_kwargs["reply_to_message_id"] = reply_to_message_id

    bot_me = await context.bot.get_me()
    annotated_path, caption_html, error = await _identify_photo_to_result(
        photo_attachment, chat_id, user=user, message_id=message_id,
        bot_username=bot_me.username)

    if error:
        await context.bot.send_message(**reply_kwargs, text=error)
        return

    with open(annotated_path, 'rb') as photo_file:
        await context.bot.send_photo(**reply_kwargs, photo=photo_file)
    await context.bot.send_message(**reply_kwargs, text=caption_html, parse_mode="HTML",
                                    disable_web_page_preview=True)
    os.unlink(annotated_path)


def _log_tracking(identifiers, all_results, merged_results, shown_matches,
                  image, image_path, processing_time_ms, chat_id, message_id, user):
    """Log identification request and matches to tracking DB. Best-effort, never raises."""
    try:
        tracker = get_tracker()
        dataset_names = [Path(ident.db.db_path).stem for ident in identifiers]
        dataset_embedders = [ident.pipeline.get_embedder_short_name() for ident in identifiers]

        num_segments = len(merged_results) if merged_results else 0
        request_id = tracker.log_request(
            telegram_user_id=user.id if user else None,
            telegram_username=user.username if user else None,
            telegram_chat_id=chat_id,
            telegram_message_id=message_id,
            image_path=image_path,
            image_width=image.width,
            image_height=image.height,
            num_segments=num_segments,
            num_datasets=len(identifiers),
            dataset_names=",".join(dataset_names),
            segmentor_model=Config.SAM3_MODEL,
            segmentor_concept=Config.DEFAULT_CONCEPT,
            processing_time_ms=processing_time_ms,
        )

        if not shown_matches:
            return

        # Build post_id→(dataset_name, embedder, rank_in_dataset) lookup from all_results
        # For each dataset's per-segment results, record the rank of each match by post_id
        post_dataset_rank = {}  # (seg_idx, post_id) → (dataset_name, embedder, rank)
        for ds_idx, ds_results in enumerate(all_results):
            ds_name = dataset_names[ds_idx]
            ds_emb = dataset_embedders[ds_idx]
            for seg_result in ds_results:
                for rank, match in enumerate(seg_result.matches):
                    key = (seg_result.segment_index, match.post_id)
                    if key not in post_dataset_rank:
                        post_dataset_rank[key] = (ds_name, ds_emb, rank)

        matches_data = []
        for seg_idx, match, rank_after_merge in shown_matches:
            seg = merged_results[seg_idx]
            key = (seg.segment_index, match.post_id)
            ds_name, ds_emb, rank_in_ds = post_dataset_rank.get(key, (None, None, None))
            matches_data.append({
                "segment_index": seg.segment_index,
                "segment_bbox": list(seg.segment_bbox),
                "segment_confidence": seg.segment_confidence,
                "dataset_name": ds_name,
                "embedder": ds_emb,
                "merge_strategy": Config.MERGE_STRATEGY,
                "character_name": match.character_name,
                "match_confidence": match.confidence,
                "match_distance": match.distance,
                "matched_post_id": match.post_id,
                "matched_source": match.source,
                "rank_in_dataset": rank_in_ds,
                "rank_after_merge": rank_after_merge,
            })
        tracker.log_matches(request_id, matches_data)
    except Exception:
        traceback.print_exc()
        print("Warning: failed to log tracking data", file=sys.stderr)


async def identify_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Identify characters in a directly sent photo."""
    try:
        await identify_and_send(context, update.effective_chat.id, update.message.effective_attachment,
                               react_message=update.message,
                               user=update.effective_user, message_id=update.message.message_id)
    except Exception as e:
        traceback.print_exc()
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Error identifying photo. Please try again.")


async def whodis(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /whodis command - identify a photo being replied to."""
    if not update.message or not update.message.reply_to_message:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Reply to a photo with /whodis to identify it."
        )
        return

    reply_to = update.message.reply_to_message
    if not reply_to.photo:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="The message you replied to doesn't contain a photo."
        )
        return

    try:
        await identify_and_send(context, update.effective_chat.id, reply_to.photo, reply_to.message_id,
                               react_message=update.message,
                               user=update.effective_user, message_id=reply_to.message_id)
    except Exception as e:
        traceback.print_exc()
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Error identifying photo. Please try again."
        )


async def fursearch(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /fursearch command.

    With args (e.g. /fursearch foxona or /fursearch this is @charactername):
        Add the photo to the database with the given character name.
        Photo can be in the replied-to message or as caption on a photo.
    Without args:
        Identify the photo being replied to.
    """
    if not update.message:
        return

    character_name = " ".join(context.args) if context.args else None

    if not character_name:
        # No name given - fall back to identify behavior
        await whodis(update, context)
        return

    if not character_name:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Please provide a character name. Example: /fursearch foxona"
        )
        return

    # Find the photo: either in the message itself (caption on photo) or in the replied-to message
    photo_message = None
    if update.message.photo:
        photo_message = update.message
    elif update.message.reply_to_message and update.message.reply_to_message.photo:
        photo_message = update.message.reply_to_message
    else:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Reply to a photo with /fursearch CharacterName, or send a photo with /fursearch CharacterName as caption."
        )
        return

    await add_photo(update, context, character_name, photo_message=photo_message)


async def reply_to_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Identify characters when replying to a photo with a bot mention, or add via character:Name."""
    user = update.effective_user
    username = f"@{user.username}" if user and user.username else (str(user.id) if user else "unknown")
    chat_id = update.effective_chat.id if update.effective_chat else "unknown"
    msg_text = update.message.text if update.message else None

    print(f"reply_to_photo: user={username} chat={chat_id} text={msg_text!r}", file=sys.stderr)

    if not update.message or not update.message.reply_to_message:
        print(f"  -> no message or reply_to_message", file=sys.stderr)
        return
    reply_to = update.message.reply_to_message
    if not reply_to.photo:
        print(f"  -> reply_to has no photo (has: text={bool(reply_to.text)}, caption={bool(reply_to.caption)}, document={bool(reply_to.document)})", file=sys.stderr)
        return

    # Check if replying with "character:Name" to add to database
    text_raw = (update.message.text or "").strip()
    char_match = CHARACTER_PATTERN.search(text_raw)
    if char_match:
        character_name = char_match.group(1)
        print(f"  -> character:{character_name} reply-to-photo, adding", file=sys.stderr)
        await add_photo(update, context, character_name, photo_message=reply_to)
        return

    text = text_raw.lower()
    bot_username = (await context.bot.get_me()).username.lower()
    print(f"  -> looking for @{bot_username} in {text!r}", file=sys.stderr)
    if f"@{bot_username}" not in text:
        print(f"  -> mention not found", file=sys.stderr)
        return

    print(f"  -> proceeding to identify photo", file=sys.stderr)

    try:
        await identify_and_send(context, update.effective_chat.id, reply_to.photo, reply_to.message_id,
                               react_message=update.message,
                               user=update.effective_user, message_id=reply_to.message_id)
    except Exception as e:
        traceback.print_exc()
        await context.bot.send_message(
            chat_id=update.effective_chat.id, reply_to_message_id=reply_to.message_id, text="Error identifying photo. Please try again."
        )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command."""
    username = update.effective_user.username
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Hi NFC! Send me a photo to find which fursuiters are in it.\n\n"
             "To make your fursuit discoverable:\n"
             "<b>Add your fursuit pictures</b> with a caption <pre>character:CharacterName</pre>\n"
             f"You can also share your telegram by captioning photos like this: <pre>character:@{username}</pre>\n\n"
             "For feedback and source code, use /info\n"
    , parse_mode="html")


async def show(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /show and /search commands - show example images of a character."""
    if not context.args:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Usage: /show CharacterName"
        )
        return

    query = " ".join(context.args)
    try:
        identifiers = get_identifiers()
        # Collect character names from all databases across all identifiers
        all_names: set[str] = set()
        all_dbs: list[Database] = []
        for ident in identifiers:
            all_dbs.append(ident.db)
            all_names.update(ident.db.get_all_character_names())

        # Try exact match first (case-insensitive)
        name_lower = {n.lower(): n for n in all_names}
        matched_names = []
        if query.lower() in name_lower:
            matched_names = [name_lower[query.lower()]]
        else:
            # Fuzzy match using difflib
            from difflib import get_close_matches
            # Match against lowercased names, map back to originals
            close = get_close_matches(query.lower(), name_lower.keys(), n=Config.DEFAULT_TOP_K, cutoff=0.5)
            matched_names = [name_lower[c] for c in close]

        if not matched_names:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"No characters found matching '{query}'."
            )
            return

        # Gather detections from all matched names across all databases
        detections = []
        for name in matched_names:
            for db in all_dbs:
                detections.extend(db.get_detections_by_character(name))

        if len(matched_names) > 1:
            names_list = ", ".join(matched_names)
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"Matched characters: {names_list}"
            )

        # Collect unique image URLs
        # When multiple characters matched (uncertainty), limit to 1 image per character
        limit_per_char = 1 if len(matched_names) > 1 else None
        seen_posts = set()
        char_image_count: dict[str, int] = {}
        # Each item is (media_source, caption, parse_mode) where media_source is a URL string or Path
        media_items: list[tuple] = []
        for det in detections:
            if det.post_id in seen_posts:
                continue
            media_source = get_source_image_url(det.source, det.post_id)
            if not media_source:
                continue
            char_key = (det.character_name or "").lower()
            if limit_per_char and char_image_count.get(char_key, 0) >= limit_per_char:
                continue
            seen_posts.add(det.post_id)
            char_image_count[char_key] = char_image_count.get(char_key, 0) + 1
            page_url = _get_page_url(det.source, det.post_id)
            safe_name = html.escape(det.character_name or "Unknown")
            safe_source = html.escape(det.source or "unknown")
            caption = f"{safe_name} ({safe_source})"
            if page_url:
                caption = f"<a href=\"{page_url}\">{safe_name}</a> ({safe_source})"
            media_items.append((media_source, caption, "HTML"))

        if not media_items:
            names_list = ", ".join(matched_names)
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"No images found for {names_list}."
            )
            return

        # Pick up to 5 random photos
        if len(media_items) > 5:
            media_items = random.sample(media_items, 5)

        # Send photos - use media group for URLs, individual sends for local files
        chat_id = update.effective_chat.id
        sent = 0
        print(f"/show: {len(media_items)} media items to send", file=sys.stderr)
        for src, cap, pm in media_items:
            try:
                if isinstance(src, Path):
                    with open(src, 'rb') as f:
                        await context.bot.send_photo(
                            chat_id=chat_id, photo=f, caption=cap, parse_mode=pm)
                else:
                    await context.bot.send_photo(
                        chat_id=chat_id, photo=src, caption=cap, parse_mode=pm)
                sent += 1
            except Exception as e:
                print(f"/show: failed to send {src!r}: {e}", file=sys.stderr)

        if sent == 0:
            names_list = ", ".join(matched_names)
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"Found {names_list} in the database, but could not load any images."
            )

    except Exception as e:
        traceback.print_exc()
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"Error looking up character: {e}"
        )


async def search(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /search command - text-based fursuit search using CLIP/SigLIP embeddings."""
    if not context.args:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Usage: /search blue fox with white markings"
        )
        return

    query = " ".join(context.args)
    try:
        identifiers = get_identifiers()
        # Only search identifiers whose embedder supports text
        text_identifiers = [
            ident for ident in identifiers
            if hasattr(ident.pipeline.embedder, "embed_text")
        ]
        if not text_identifiers:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="Text search is not available. No datasets use a text-capable embedder (CLIP/SigLIP)."
            )
            return

        # Merge text search results from all text-capable identifiers
        def _run_search():
            results = []
            for ident in text_identifiers:
                results.extend(ident.search_text(query, top_k=10))
            return results
        all_results = await asyncio.to_thread(_run_search)

        if not all_results:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"No matches found for '{query}'."
            )
            return

        # Deduplicate by character, keeping best match
        seen = {}
        for r in all_results:
            name = r.character_name or "unknown"
            if name not in seen or r.confidence > seen[name].confidence:
                seen[name] = r
        top_matches = sorted(seen.values(), key=lambda x: x.confidence, reverse=True)[:5]
        print(f'Found top matches: {top_matches}')

        lines = [f"Search results for '<b>{html.escape(query)}</b>':"]
        for i, m in enumerate(top_matches, 1):
            name = html.escape(m.character_name or "Unknown")
            url = _get_page_url(m.source, m.post_id)
            if url:
                lines.append(f"  {i}. <a href=\"{url}\">{name}</a> ({m.confidence*100:.1f}%)")
            else:
                lines.append(f"  {i}. {name} ({m.confidence*100:.1f}%)")

        # Send one example image per top match character
        media = []
        for m in top_matches:
            # Find one linkable image for this character
            for ident in identifiers:
                found = False
                for det in ident.db.get_detections_by_character(m.character_name):
                    img_url = get_source_image_url(det.source, det.post_id)
                    if not img_url:
                        continue
                    page_url = _get_page_url(det.source, det.post_id)
                    safe_name = html.escape(det.character_name or "Unknown")
                    safe_source = html.escape(det.source or "unknown")
                    caption = f"{safe_name} ({safe_source})"
                    if page_url:
                        caption = f"<a href=\"{page_url}\">{safe_name}</a> ({safe_source})"
                    media.append(InputMediaPhoto(media=img_url, caption=caption, parse_mode="HTML"))
                    found = True
                    break
                if found:
                    break

        msg = "\n".join(lines)
        await context.bot.send_message(
            chat_id=update.effective_chat.id, text=msg, parse_mode="HTML",
            disable_web_page_preview=True)

        # Try media group first, fall back to individual photos
        if media:
            try:
                if len(media) >= 2:
                    await context.bot.send_media_group(
                        chat_id=update.effective_chat.id, media=media)
                else:
                    await context.bot.send_photo(
                        chat_id=update.effective_chat.id,
                        photo=media[0].media, caption=media[0].caption,
                        parse_mode=media[0].parse_mode)
            except Exception:
                for item in media:
                    try:
                        await context.bot.send_photo(
                            chat_id=update.effective_chat.id,
                            photo=item.media, caption=item.caption,
                            parse_mode=item.parse_mode)
                    except Exception:
                        continue
    except Exception as e:
        traceback.print_exc()
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"Error performing search: {e}"
        )


async def adminstats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /adminstats command - detailed stats for authorized users only."""
    if not aitool.is_user_authorized(update):
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="You are not authorized to use this command."
        )
        return

    try:
        from collections import Counter
        from sam3_fursearch.storage.database import bucketize

        identifiers = get_identifiers()
        all_characters = set()
        all_posts = set()
        stats_list = [ident.get_stats() for ident in identifiers]
        combined_stats = FursuitIdentifier.get_combined_stats(stats_list)

        char_posts = Counter()
        post_segments = Counter()
        for ident in identifiers:
            all_characters.update(ident.db.get_all_character_names())
            all_posts.update(ident.db.get_all_post_ids())
            for char, cnt in ident.db.get_character_post_counts().items():
                char_posts[char] += cnt
            for post, cnt in ident.db.get_post_segment_counts().items():
                post_segments[post] += cnt

        combined_stats["unique_characters"] = len(all_characters)
        combined_stats["unique_posts"] = len(all_posts)
        combined_stats["posts_per_character"] = bucketize(Counter(char_posts.values()))
        combined_stats["segments_per_post"] = bucketize(Counter(post_segments.values()))
        del combined_stats["top_characters"]

        identifiers = get_identifiers()

        char_posts = Counter()
        for ident in identifiers:
            if Path(ident.db.db_path).stem != Config.DEFAULT_DATASET:
                continue
            for char, cnt in ident.db.get_character_post_counts().items():
                char_posts[char] += cnt

        top_chars = char_posts.most_common(10)
        total_chars = len(char_posts)

        sender_counts = Counter()
        for ident in identifiers:
            conn = ident.db._connect()
            c = conn.cursor()
            c.execute("""
                SELECT uploaded_by, COUNT(DISTINCT post_id)
                FROM detections
                WHERE source = 'tgbot' AND uploaded_by IS NOT NULL
                GROUP BY uploaded_by
            """)
            for sender, cnt in c.fetchall():
                sender_counts[sender] += cnt

        top_senders = sender_counts.most_common(10)
        combined_stats["top_characters"] = [{"character_name": char, "post_count": cnt} for char, cnt in top_chars]
        combined_stats["top_senders"] = [{"sender": sender, "post_count": cnt} for sender, cnt in top_senders]
        combined_stats["total_characters"] = total_chars

        # Tracking info
        try:
            tracker = get_tracker()
            combined_stats["tracking"] = tracker.get_stats()
        except Exception:
            pass

        import json
        msg = json.dumps(combined_stats, indent=2, ensure_ascii=False, default=str)
        await context.bot.send_message(chat_id=update.effective_chat.id, text=msg)

    except Exception as e:
        traceback.print_exc()
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"Error getting stats: {e}"
        )


async def info(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /info command - Show bot info"""
    try:
        end = time.perf_counter()
        uptime = f"{(end - start_time) / 3600 :.1f} hours"
        version = get_git_version()
        lines = [
            "https://github.com/fursearch/fursearch",
            "Send feedback: /feedback your message",
            f"Version: {version}",
            f"Uptime: {uptime}",
        ]
        msg = "\n".join(lines)
        await context.bot.send_message(
            chat_id=update.effective_chat.id, text=msg, parse_mode="HTML",
            disable_web_page_preview=True)

    except Exception as e:
        traceback.print_exc()
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"Error sending info: {e}"
        )


async def restart(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /restart command - restart the bot process."""
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Restarting bot..."
    )

    # Stop the application gracefully
    context.application.stop_running()

    # Replace current process with a new instance
    os.execv(sys.executable, [sys.executable] + sys.argv)


async def debug_all_messages(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Debug handler to log all incoming messages."""
    msg = update.message
    if not msg:
        return
    user = update.effective_user
    username = f"@{user.username}" if user and user.username else (str(user.id) if user else "unknown")
    print(f"DEBUG incoming: user={username} chat={update.effective_chat.id}", file=sys.stderr)
    print(f"  text={msg.text!r} caption={msg.caption!r}", file=sys.stderr)
    print(f"  photo={bool(msg.photo)} reply_to={bool(msg.reply_to_message)}", file=sys.stderr)
    if msg.reply_to_message:
        print(f"  reply_to.photo={bool(msg.reply_to_message.photo)}", file=sys.stderr)

    # Cache admin's chat_id whenever they message the bot
def _get_admin_chat_id() -> Optional[int]:
    """Return the configured admin chat_id from the environment, or None."""
    raw = os.environ.get("FURSEARCH_ADMIN_CHAT_ID", "").strip()
    return int(raw) if raw else None


async def feedback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /feedback <message> - forward a message to the admin."""
    if not update.message:
        return

    message_text = " ".join(context.args) if context.args else ""
    if not message_text.strip():
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Usage: /feedback <your message>\n\nYour feedback will be forwarded to bot admins."
        )
        return

    admin_chat_id = _get_admin_chat_id()
    if not admin_chat_id:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Feedback is not configured on this bot. Sorry!"
        )
        return

    user = update.effective_user
    tracker = get_tracker()
    feedback_id = tracker.log_user_feedback(
        telegram_user_id=user.id if user else None,
        telegram_chat_id=update.effective_chat.id,
        telegram_username=user.username if user else None,
        message=message_text,
    )

    try:
        sent = await context.bot.send_message(
            chat_id=admin_chat_id,
            text=f"📬 <b>Feedback #{feedback_id}</b>\n\n{html.escape(message_text)}\n\n"
                 f"<i>Reply to this message to respond to the sender.</i>",
            parse_mode="HTML",
        )
        tracker.set_feedback_admin_message_id(feedback_id, sent.message_id)
    except Exception:
        traceback.print_exc()
        print(f"Warning: could not deliver feedback #{feedback_id} to admin", file=sys.stderr)

    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Your feedback has been sent. Thank you!"
    )


async def admin_reply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Route admin replies to feedback messages back to the original sender."""
    if not update.message or not update.message.reply_to_message:
        return

    admin_chat_id = _get_admin_chat_id()
    if not admin_chat_id:
        return

    if update.effective_chat.id != admin_chat_id:
        return

    replied_msg_id = update.message.reply_to_message.message_id
    tracker = get_tracker()
    fb = tracker.get_feedback_by_admin_message_id(replied_msg_id)
    if fb is None:
        return  # Not a reply to a feedback message

    reply_text = update.message.text or update.message.caption or ""
    if not reply_text.strip():
        return

    try:
        sent = await context.bot.send_message(
            chat_id=fb["telegram_chat_id"],
            text=f"💬 <b>Message from the bot admin:</b>\n\n{html.escape(reply_text)}",
            parse_mode="HTML",
        )
        tracker.add_feedback_thread_message(fb["id"], user_message_id=sent.message_id)
        await _react(update.message, "✅")
    except Exception:
        traceback.print_exc()
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Failed to deliver reply to the user."
        )


async def user_feedback_reply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Forward a user's reply to a bot feedback message back to the admin."""
    if not update.message or not update.message.reply_to_message:
        return

    replied_msg_id = update.message.reply_to_message.message_id
    user_chat_id = update.effective_chat.id
    tracker = get_tracker()
    fb = tracker.get_feedback_by_user_message_id(user_chat_id, replied_msg_id)
    if fb is None:
        return

    reply_text = update.message.text or update.message.caption or ""
    if not reply_text.strip():
        return

    admin_chat_id = _get_admin_chat_id()
    if not admin_chat_id:
        return

    try:
        sent = await context.bot.send_message(
            chat_id=admin_chat_id,
            reply_to_message_id=fb["admin_message_id"],
            text=f"↩️ <b>Reply (feedback #{fb['id']}):</b>\n\n{html.escape(reply_text)}",
            parse_mode="HTML",
        )
        tracker.add_feedback_thread_message(fb["id"], admin_message_id=sent.message_id)
        await _react(update.message, "✅")
    except Exception:
        traceback.print_exc()


def build_application(token: str):
    """Create a Telegram application with all handlers."""
    app = ApplicationBuilder().token(token).concurrent_updates(True).build()
    # Debug handler - logs all messages (group=-1 runs before other handlers)
    app.add_handler(MessageHandler(filters.ALL, debug_all_messages), group=-1)
    app.add_handler(MessageHandler((~filters.COMMAND) & filters.PHOTO, photo))
    app.add_handler(MessageHandler((~filters.COMMAND) & filters.TEXT & filters.REPLY, reply_to_photo))
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("info", info))
    app.add_handler(CommandHandler("adminstats", adminstats))
    app.add_handler(CommandHandler("show", show))
    app.add_handler(CommandHandler("search", search))
    app.add_handler(CommandHandler("whodis", whodis))
    app.add_handler(CommandHandler("fursearch", fursearch))
    app.add_handler(CommandHandler("feedback", feedback))
    app.add_handler(CommandHandler("aitool", aitool.handle_aitool))
    app.add_handler(CommandHandler("restart", restart))
    app.add_handler(CommandHandler("commit", aitool.handle_commit))
    app.add_handler(MessageHandler(filters.TEXT & filters.REPLY, admin_reply), group=1)
    app.add_handler(MessageHandler((~filters.COMMAND) & filters.TEXT & filters.REPLY, user_feedback_reply), group=2)
    return app


async def run_bot_and_web():
    """Run multiple Telegram bots and web server concurrently."""
    tokens_str = os.environ.get("TG_BOT_TOKENS", os.environ.get("TG_BOT_TOKEN", ""))

    tokens = [t.strip() for t in tokens_str.split(",") if t.strip()]

    if not tokens:
        print("Error: TG_BOT_TOKEN or TG_BOT_TOKENS not set", file=sys.stderr)
        sys.exit(1)

    applications = [build_application(token) for token in tokens]

    print("Starting {len(applications)} bot(s)...")

    for app in applications:
        await app.initialize()
        await app.start()
        await app.updater.start_polling()
        me = await app.bot.get_me()
        print(f"  @{me.username} running")

    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass
    finally:
        for app in applications:
            await app.updater.stop()
            await app.stop()
            await app.shutdown()
        await web_runner.cleanup()


if __name__ == "__main__":
    asyncio.run(run_bot_and_web())
